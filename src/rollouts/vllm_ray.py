from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _split_visible_devices(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _normalize_visible_devices_csv(value: str) -> str:
    devices = _split_visible_devices(value)
    if not devices:
        return ""

    normalized: list[str] = []
    unresolved: list[str] = []
    for token in devices:
        try:
            normalized.append(str(int(token)))
        except ValueError:
            unresolved.append(token)

    if not unresolved:
        return ",".join(normalized)

    try:
        import pynvml
    except Exception as exc:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES contains GPU UUIDs, but pynvml is unavailable to map them "
            f"for vLLM: {unresolved}"
        ) from exc

    try:
        pynvml.nvmlInit()
        uuid_to_index: dict[str, int] = {}
        device_count = pynvml.nvmlDeviceGetCount()
        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            uuid_to_index[str(uuid)] = idx

        resolved: list[str] = []
        for token in devices:
            try:
                resolved.append(str(int(token)))
                continue
            except ValueError:
                pass

            matched_index = None
            for uuid, idx in uuid_to_index.items():
                if uuid == token or uuid.startswith(token):
                    matched_index = idx
                    break
            if matched_index is None:
                raise RuntimeError(
                    "Could not map CUDA_VISIBLE_DEVICES entry to a physical GPU index for vLLM: "
                    f"{token}"
                )
            resolved.append(str(matched_index))
        return ",".join(resolved)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_object(value: Any) -> list[Any]:
    if not _dist_is_ready() or dist.get_world_size() <= 1:
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_from_rank0(value: Any) -> Any:
    if not _dist_is_ready() or dist.get_world_size() <= 1:
        return value
    payload = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _flatten(groups: list[list[str]]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return merged


def _split_flat(items: list[str], counts: list[int]) -> list[list[str]]:
    chunks: list[list[str]] = []
    cursor = 0
    for count in counts:
        chunks.append(items[cursor : cursor + count])
        cursor += count
    if cursor != len(items):
        raise RuntimeError(f"rollout split mismatch: consumed={cursor} total={len(items)}")
    return chunks


def _chunk_prompt_groups_by_tokens(
    prompt_groups: list[list[str]],
    tokenizer,
    max_batch_prompts: int,
    max_batch_tokens: int,
    output_tokens_per_prompt: int,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for group in prompt_groups:
        for prompt in group:
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompt_tokens = max(prompt_tokens + max(output_tokens_per_prompt, 0), 1)
            would_overflow_prompts = len(current_chunk) >= max_batch_prompts
            would_overflow_tokens = current_chunk and (current_tokens + prompt_tokens > max_batch_tokens)
            if would_overflow_prompts or would_overflow_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(prompt)
            current_tokens += prompt_tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _get_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidate = str(sock.getsockname()[0])
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass

    try:
        candidate = socket.gethostbyname(socket.gethostname())
        if candidate and not str(candidate).startswith("127."):
            return str(candidate)
    except OSError:
        pass

    return "127.0.0.1"


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _unwrap_engine_model(engine):
    return engine.module if hasattr(engine, "module") else engine


def _http_timeout_seconds() -> float:
    return max(30.0, float(os.environ.get("PAR_VLLM_HTTP_TIMEOUT_SECONDS", "300")))


def _server_startup_timeout_seconds() -> float:
    return max(60.0, float(os.environ.get("PAR_VLLM_SERVER_STARTUP_TIMEOUT_SECONDS", "900")))


def _resolve_rollout_devices(config) -> str:
    if getattr(config, "rollout_visible_devices", None):
        return _normalize_visible_devices_csv(str(config.rollout_visible_devices))

    requested = int(getattr(config, "rollout_vllm_tensor_parallel_size", 1))
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value:
        devices = _split_visible_devices(_normalize_visible_devices_csv(env_value))
    else:
        devices = [str(idx) for idx in range(torch.cuda.device_count())]

    if len(devices) < requested:
        raise RuntimeError(
            f"Not enough visible GPUs for vLLM rollout: requested {requested}, found {len(devices)}."
        )
    return ",".join(devices[-requested:])


class _ThreadResult:
    def __init__(self) -> None:
        self.error: str | None = None
        self.value: Any = None


class _ServerState:
    def __init__(self, index: int, device: str, run_dir: Path) -> None:
        self.index = index
        self.device = str(device)
        self.port = _get_open_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = run_dir / f"vllm_rollout_server_{index}.log"
        self.process: subprocess.Popen[str] | None = None
        self.openai_client = None
        self.trainer_group = None


class VLLMRayRolloutBackend:
    """GRPO rollout backend implemented via one TP=1 vLLM HTTP server per rollout GPU.

    The config name remains `vllm_ray` for compatibility with the launcher. The
    earlier TP>1 single-server design kept hanging during vLLM's internal
    multiprocess bootstrap on the cluster. This implementation avoids that path
    entirely by launching independent single-GPU servers and sharding prompts
    across them.
    """

    def __init__(self, config, tokenizer, policy_engine, log_fn):
        self.config = config
        self.tokenizer = tokenizer
        self.policy_engine = policy_engine
        self.log_fn = log_fn
        self.global_rank = int(config.global_rank)
        self.world_size = int(config.world_size)
        self.visible_devices = _resolve_rollout_devices(config)
        self.rollout_devices = _split_visible_devices(self.visible_devices)
        if not self.rollout_devices:
            raise RuntimeError("No rollout GPUs were resolved for the vLLM backend.")

        self.server_count = len(self.rollout_devices)
        self.max_batch_prompts = max(1, int(config.rollout_max_batch_prompts))
        self.max_batch_tokens = max(1, int(config.rollout_max_batch_tokens))
        self.sync_interval_steps = max(1, int(config.rollout_sync_interval_steps))
        self.http_timeout_seconds = _http_timeout_seconds()
        self.server_startup_timeout_seconds = _server_startup_timeout_seconds()
        self.last_synced_step = -1

        self._requests = None
        self._servers: list[_ServerState] = []
        self._weight_metadata = None
        self._trainer_send_weights_args_cls = None
        self._trainer_weight_engine_cls = None
        self._served_model = self.config.model_path

        status = None
        if self.global_rank == 0:
            try:
                self._initialize_rank0()
                status = {
                    "ok": True,
                    "visible_devices": self.visible_devices,
                    "server_count": self.server_count,
                    "base_urls": [server.base_url for server in self._servers],
                }
            except Exception:
                status = {"ok": False, "error": traceback.format_exc()}

        status = _broadcast_from_rank0(status)
        if not status["ok"]:
            raise RuntimeError(
                "Failed to initialize the vLLM rollout backend.\n"
                f"{status['error']}"
            )

        if self.global_rank == 0:
            self.log_fn(
                "[rollout] enabled backend=vllm_ray implementation=http_nccl_per_gpu "
                f"visible_devices={status['visible_devices']} "
                f"servers={status['server_count']} "
                f"max_batch_prompts={self.max_batch_prompts} "
                f"max_batch_tokens={self.max_batch_tokens} "
                f"server_base_urls={status['base_urls']}"
            )

    def _tail_server_log(self, server: _ServerState, max_lines: int = 120) -> str:
        if not server.log_path.exists():
            return f"<no vLLM server log available for server {server.index}>"
        try:
            lines = server.log_path.read_text(errors="replace").splitlines()
        except Exception as exc:
            return f"<failed to read vLLM server {server.index} log: {exc}>"
        if not lines:
            return f"<vLLM server {server.index} log is empty>"
        tail = "\n".join(lines[-max_lines:])
        return (
            f"Last {min(max_lines, len(lines))} line(s) of vLLM server {server.index} log:\n"
            f"{tail}"
        )

    def _ensure_server_running(self, server: _ServerState) -> None:
        if server.process is None:
            raise RuntimeError(f"vLLM rollout server {server.index} was not started.")
        exit_code = server.process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"vLLM rollout server {server.index} exited unexpectedly with code {exit_code}.\n"
                f"{self._tail_server_log(server)}"
            )

    def _vllm_executable(self) -> str:
        binary = shutil.which("vllm")
        if binary:
            return binary
        raise RuntimeError(
            "Could not find the `vllm` executable in PATH. "
            "The GRPO rollout backend launches `vllm serve` directly."
        )

    def _build_vllm_serve_command(
        self, server: _ServerState, include_optional_flags: bool
    ) -> list[str]:
        binary = self._vllm_executable()
        command = [
            binary,
            "serve",
            self.config.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(server.port),
            "--tensor-parallel-size",
            "1",
            "--dtype",
            str(self.config.policy_dtype),
            "--gpu-memory-utilization",
            str(float(self.config.rollout_gpu_memory_utilization)),
            "--max-model-len",
            str(int(self.config.rollout_max_model_len)),
            "--weight-transfer-config",
            json.dumps({"backend": str(self.config.rollout_weight_transfer_backend)}),
        ]
        if include_optional_flags:
            command.append("--trust-remote-code")
        if bool(self.config.rollout_enforce_eager):
            command.append("--enforce-eager")
        if bool(self.config.rollout_init_with_dummy_weights):
            command.extend(["--load-format", "dummy"])
        return command

    def _spawn_server_process(self, server: _ServerState, command: list[str]) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = server.device
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env.pop("VLLM_HOST_IP", None)
        env.pop("VLLM_PORT", None)
        if "TMPDIR" in os.environ:
            env["TMPDIR"] = os.environ["TMPDIR"]
        if "TEMP" in os.environ:
            env["TEMP"] = os.environ["TEMP"]
        if "TMP" in os.environ:
            env["TMP"] = os.environ["TMP"]
        if "TRITON_CACHE_DIR" in os.environ:
            env["TRITON_CACHE_DIR"] = os.environ["TRITON_CACHE_DIR"]

        with server.log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("[launcher] " + " ".join(command) + "\n")

        log_file = server.log_path.open("a", encoding="utf-8")
        try:
            server.process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
        finally:
            log_file.close()

    def _launch_servers(self) -> None:
        run_dir = Path(self.config.local_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._servers = []

        for index, device in enumerate(self.rollout_devices):
            server = _ServerState(index=index, device=device, run_dir=run_dir)
            self.log_fn(
                f"[rollout] launching vLLM server idx={server.index} "
                f"device={server.device} base_url={server.base_url} "
                f"log_file={server.log_path}"
            )
            self._spawn_server_process(
                server,
                self._build_vllm_serve_command(server, include_optional_flags=True),
            )
            self._servers.append(server)

    def _http_get_json(
        self, server: _ServerState, path: str, timeout: float | None = None
    ) -> dict[str, Any]:
        self._ensure_server_running(server)
        assert self._requests is not None
        response = self._requests.get(
            f"{server.base_url}{path}",
            timeout=timeout or self.http_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _http_post_json(
        self,
        server: _ServerState,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_server_running(server)
        assert self._requests is not None
        response = self._requests.post(
            f"{server.base_url}{path}",
            **({"json": payload} if payload is not None else {}),
            timeout=timeout or self.http_timeout_seconds,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def _wait_until_server_ready(self, server: _ServerState) -> int:
        deadline = time.time() + self.server_startup_timeout_seconds
        last_error = ""
        while time.time() < deadline:
            if server.process is not None and server.process.poll() is not None:
                log_tail = self._tail_server_log(server)
                if "unrecognized arguments:" in log_tail and "--trust-remote-code" in log_tail:
                    self.log_fn(
                        f"[rollout] retrying vLLM server {server.index} startup without "
                        "optional CLI flags after an unrecognized-arguments failure"
                    )
                    server.process = None
                    self._spawn_server_process(
                        server,
                        self._build_vllm_serve_command(server, include_optional_flags=False),
                    )
                    time.sleep(2.0)
                    continue
                raise RuntimeError(
                    f"vLLM rollout server {server.index} exited during startup with code "
                    f"{server.process.returncode}.\n{log_tail}"
                )
            try:
                payload = self._http_get_json(server, "/get_world_size", timeout=5.0)
                world_size = int(payload["world_size"])
                if world_size <= 0:
                    raise RuntimeError(
                        f"Invalid rollout server {server.index} world size: {world_size}"
                    )
                self.log_fn(
                    f"[rollout] vLLM server {server.index} is ready "
                    f"world_size={world_size} base_url={server.base_url}"
                )
                return world_size
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2.0)
        raise RuntimeError(
            f"Timed out waiting for vLLM rollout server {server.index} to become ready.\n"
            f"Last observed error: {last_error}\n{self._tail_server_log(server)}"
        )

    def _thread_http_init_weight_transfer_engine(
        self,
        server: _ServerState,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        result: _ThreadResult,
    ) -> None:
        try:
            self._http_post_json(
                server,
                "/init_weight_transfer_engine",
                {
                    "init_info": {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": rank_offset,
                        "world_size": world_size,
                    }
                },
                timeout=self.http_timeout_seconds,
            )
        except Exception:
            result.error = traceback.format_exc()

    def _thread_http_update_weights(
        self,
        server: _ServerState,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        packed: bool,
        result: _ThreadResult,
    ) -> None:
        try:
            self._http_post_json(
                server,
                "/update_weights",
                {
                    "update_info": {
                        "names": names,
                        "dtype_names": dtype_names,
                        "shapes": shapes,
                        "packed": packed,
                    }
                },
                timeout=self.http_timeout_seconds,
            )
        except Exception:
            result.error = traceback.format_exc()

    def _pause_generation(self, server: _ServerState) -> None:
        self._http_post_json(server, "/pause", payload=None, timeout=self.http_timeout_seconds)

    def _resume_generation(self, server: _ServerState) -> None:
        self._http_post_json(server, "/resume", payload=None, timeout=self.http_timeout_seconds)

    def _initialize_weight_transfer_for_server(self, server: _ServerState) -> None:
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )

        self._trainer_send_weights_args_cls = NCCLTrainerSendWeightsArgs
        self._trainer_weight_engine_cls = NCCLWeightTransferEngine

        world_size = 2
        master_address = _get_host_ip()
        master_port = _get_open_port()
        self.log_fn(
            f"[rollout] initializing NCCL weight transfer for server {server.index} "
            f"trainer+server world_size={world_size}"
        )
        result = _ThreadResult()
        init_thread = threading.Thread(
            target=self._thread_http_init_weight_transfer_engine,
            args=(server, master_address, master_port, 1, world_size, result),
            daemon=True,
        )
        init_thread.start()
        server.trainer_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_thread.join(timeout=self.http_timeout_seconds)
        if init_thread.is_alive():
            raise RuntimeError(
                f"Timed out waiting for vLLM server {server.index} to initialize weight transfer.\n"
                f"{self._tail_server_log(server)}"
            )
        if result.error:
            raise RuntimeError(
                f"vLLM server {server.index} failed during weight-transfer initialization.\n"
                f"{result.error}\n{self._tail_server_log(server)}"
            )
        self.log_fn(f"[rollout] vLLM server {server.index} weight-transfer engine initialized")

    def _initialize_rank0(self) -> None:
        import requests
        from openai import OpenAI

        if self.config.use_lora:
            raise RuntimeError("rollout_backend=vllm_ray does not support LoRA policy models yet.")

        self._requests = requests.Session()
        self._launch_servers()

        for server in self._servers:
            inference_world_size = self._wait_until_server_ready(server)
            if inference_world_size != 1:
                raise RuntimeError(
                    f"vLLM rollout server {server.index} reported unexpected world size "
                    f"{inference_world_size}; expected 1.\n{self._tail_server_log(server)}"
                )
            server.openai_client = OpenAI(base_url=f"{server.base_url}/v1", api_key="EMPTY")
            self._initialize_weight_transfer_for_server(server)

        self._weight_metadata = self._collect_weight_metadata()

    def _collect_weight_metadata(self) -> tuple[list[str], list[str], list[list[int]]]:
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, param in _unwrap_engine_model(self.policy_engine).named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))
        return names, dtype_names, shapes

    def _named_parameters(self):
        return _unwrap_engine_model(self.policy_engine).named_parameters()

    def _sampling_config(self, generations_per_prompt: int, do_sample: bool) -> dict[str, Any]:
        config = {
            "n": int(generations_per_prompt),
            "max_tokens": int(self.config.max_new_tokens),
            "skip_special_tokens": True,
        }
        if getattr(self.config, "use_beam_search", False):
            config.update(
                {
                    "use_beam_search": True,
                    "best_of": max(int(self.config.num_beams), int(generations_per_prompt)),
                    "early_stopping": self.config.early_stopping,
                    "temperature": 0.0,
                }
            )
        elif do_sample:
            config.update(
                {
                    "temperature": float(self.config.temperature),
                    "top_p": float(self.config.top_p),
                    "top_k": int(self.config.top_k),
                }
            )
        else:
            config.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
        return config

    def _maybe_sync_weights(self, step: int) -> None:
        status = None
        if self.global_rank == 0:
            try:
                should_sync = (
                    self.last_synced_step < 0
                    or (step - self.last_synced_step) >= self.sync_interval_steps
                )
                if should_sync:
                    start_time = time.time()
                    names, dtype_names, shapes = self._weight_metadata
                    packed = bool(self.config.rollout_use_packed_weight_transfer)
                    for server in self._servers:
                        self.log_fn(
                            f"[rollout] syncing policy weights to vLLM server {server.index} "
                            f"at step {step}"
                        )
                        resumed = False
                        try:
                            self._pause_generation(server)
                            result = _ThreadResult()
                            update_thread = threading.Thread(
                                target=self._thread_http_update_weights,
                                args=(server, names, dtype_names, shapes, packed, result),
                                daemon=True,
                            )
                            update_thread.start()
                            trainer_args = self._trainer_send_weights_args_cls(
                                group=server.trainer_group,
                                packed=packed,
                            )
                            self._trainer_weight_engine_cls.trainer_send_weights(
                                iterator=self._named_parameters(),
                                trainer_args=trainer_args,
                            )
                            update_thread.join(timeout=self.http_timeout_seconds)
                            if update_thread.is_alive():
                                raise RuntimeError(
                                    f"Timed out waiting for vLLM server {server.index} to finish "
                                    f"/update_weights.\n{self._tail_server_log(server)}"
                                )
                            if result.error:
                                raise RuntimeError(
                                    f"vLLM server {server.index} failed during /update_weights.\n"
                                    f"{result.error}\n{self._tail_server_log(server)}"
                                )
                            self._resume_generation(server)
                            resumed = True
                        finally:
                            if not resumed:
                                try:
                                    self._resume_generation(server)
                                except Exception:
                                    pass

                    self.last_synced_step = int(step)
                    status = {
                        "ok": True,
                        "synced": True,
                        "step": self.last_synced_step,
                        "elapsed": time.time() - start_time,
                    }
                else:
                    status = {"ok": True, "synced": False, "step": self.last_synced_step}
            except Exception:
                status = {"ok": False, "error": traceback.format_exc()}

        status = _broadcast_from_rank0(status)
        if not status["ok"]:
            raise RuntimeError(
                "Failed to synchronize policy weights into the vLLM rollout engine.\n"
                f"{status['error']}"
            )
        if status.get("synced") and self.global_rank == 0:
            self.log_fn(
                f"[rollout] synced policy weights to all vLLM servers at step {status['step']} "
                f"in {status['elapsed']:.2f}s"
            )

    def _generate_for_prompt(
        self, server: _ServerState, prompt: str, sampling_config: dict[str, Any]
    ) -> list[str]:
        assert server.openai_client is not None

        extra_body = {"skip_special_tokens": bool(sampling_config.get("skip_special_tokens", True))}
        if "top_k" in sampling_config:
            extra_body["top_k"] = int(sampling_config["top_k"])
        if sampling_config.get("use_beam_search"):
            extra_body["use_beam_search"] = True
            extra_body["best_of"] = int(sampling_config["best_of"])
            extra_body["early_stopping"] = sampling_config["early_stopping"]

        response = server.openai_client.completions.create(
            model=self._served_model,
            prompt=prompt,
            n=int(sampling_config["n"]),
            max_tokens=int(sampling_config["max_tokens"]),
            temperature=float(sampling_config["temperature"]),
            top_p=float(sampling_config["top_p"]),
            timeout=self.http_timeout_seconds,
            extra_body=extra_body,
        )
        return [choice.text for choice in response.choices]

    def _thread_generate_chunk(
        self,
        server: _ServerState,
        assignments: list[tuple[int, str]],
        sampling_config: dict[str, Any],
        result: _ThreadResult,
    ) -> None:
        try:
            outputs: list[tuple[int, list[str]]] = []
            for local_idx, prompt in assignments:
                outputs.append((local_idx, self._generate_for_prompt(server, prompt, sampling_config)))
            result.value = outputs
        except Exception:
            result.error = traceback.format_exc()

    def sample(self, prompts: list[str], generations_per_prompt: int, do_sample: bool, step: int) -> list[str]:
        self._maybe_sync_weights(step)

        payload = None
        prompt_groups = [request["prompts"] for request in _all_gather_object({"prompts": list(prompts)})]
        if self.global_rank == 0:
            try:
                all_prompts = _flatten(prompt_groups)
                flat_outputs: list[str] = []
                if all_prompts:
                    sampling_config = self._sampling_config(generations_per_prompt, do_sample)
                    prompt_chunks = _chunk_prompt_groups_by_tokens(
                        prompt_groups,
                        self.tokenizer,
                        self.max_batch_prompts,
                        self.max_batch_tokens,
                        int(self.config.max_new_tokens) * max(1, int(generations_per_prompt)),
                    )
                    total_prompts = sum(len(chunk) for chunk in prompt_chunks)
                    self.log_fn(
                        f"[rollout] generating via vLLM servers prompts={total_prompts} "
                        f"chunks={len(prompt_chunks)} servers={len(self._servers)}"
                    )
                    for chunk_idx, prompt_chunk in enumerate(prompt_chunks, start=1):
                        self.log_fn(
                            f"[rollout] generating chunk {chunk_idx}/{len(prompt_chunks)} "
                            f"prompts={len(prompt_chunk)}"
                        )
                        assignments: list[list[tuple[int, str]]] = [[] for _ in self._servers]
                        for local_idx, prompt in enumerate(prompt_chunk):
                            assignments[local_idx % len(self._servers)].append((local_idx, prompt))

                        threads: list[threading.Thread] = []
                        results: list[_ThreadResult] = []
                        for server, server_assignments in zip(self._servers, assignments):
                            result = _ThreadResult()
                            results.append(result)
                            if not server_assignments:
                                result.value = []
                                continue
                            thread = threading.Thread(
                                target=self._thread_generate_chunk,
                                args=(server, server_assignments, sampling_config, result),
                                daemon=True,
                            )
                            thread.start()
                            threads.append(thread)

                        for thread in threads:
                            thread.join(timeout=self.http_timeout_seconds)
                        merged_outputs: list[tuple[int, list[str]]] = []
                        active_threads = [thread for thread in threads if thread.is_alive()]
                        if active_threads:
                            raise RuntimeError(
                                f"Timed out waiting for {len(active_threads)} vLLM generation "
                                f"thread(s) to finish."
                            )
                        for server, result in zip(self._servers, results):
                            if result.error:
                                raise RuntimeError(
                                    f"vLLM rollout server {server.index} failed during generation.\n"
                                    f"{result.error}\n{self._tail_server_log(server)}"
                                )
                            merged_outputs.extend(result.value or [])

                        for _, texts in sorted(merged_outputs, key=lambda item: item[0]):
                            flat_outputs.extend(texts)

                counts = [len(group) * int(generations_per_prompt) for group in prompt_groups]
                payload = {"ok": True, "chunks": _split_flat(flat_outputs, counts)}
            except Exception:
                payload = {"ok": False, "error": traceback.format_exc()}

        payload = _broadcast_from_rank0(payload)
        if not payload["ok"]:
            raise RuntimeError(
                "vLLM rollout generation failed.\n"
                f"{payload['error']}"
            )

        rank = dist.get_rank() if _dist_is_ready() else 0
        return payload["chunks"][rank]

    def shutdown(self) -> None:
        if self.global_rank != 0:
            return

        try:
            if self._requests is not None:
                self._requests.close()
        except Exception:
            pass
        finally:
            self._requests = None

        for server in self._servers:
            try:
                if server.process is not None and server.process.poll() is None:
                    server.process.terminate()
                    try:
                        server.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.process.kill()
                        server.process.wait(timeout=30)
            except Exception as exc:
                self.log_fn(f"[rollout] failed to stop vLLM server {server.index} cleanly: {exc}")
            finally:
                server.process = None
                server.trainer_group = None
                server.openai_client = None

        self._servers = []

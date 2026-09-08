from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency in some environments
    pynvml = None

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
SRC_ROOT_STR = str(SRC_ROOT)
if SRC_ROOT_STR in sys.path:
    sys.path.remove(SRC_ROOT_STR)
sys.path.insert(0, SRC_ROOT_STR)
existing_pythonpath = [
    path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if path and path != SRC_ROOT_STR
]
os.environ["PYTHONPATH"] = os.pathsep.join([SRC_ROOT_STR, *existing_pythonpath])

from configs.config import DATASET_PRESETS  # noqa: E402
from dataloaders.dataset import (  # noqa: E402
    canonical_split_path,
    canonicalize_preference_row,
    dataset_dir_name,
    dataset_split_path,
    format_canonical_prompt,
)
from models.models import (  # noqa: E402
    AutoModelForCausalLMWithScalarHead,
    AutoModelForCausalLMWithScalarHeadODIN,
)


def normalize_visible_devices_csv(value: str) -> str:
    devices = [token.strip() for token in str(value).split(",") if token.strip()]
    if not devices:
        return ""

    normalized: List[str] = []
    unresolved: List[str] = []
    for token in devices:
        try:
            normalized.append(str(int(token)))
        except ValueError:
            unresolved.append(token)

    if not unresolved:
        return ",".join(normalized)

    if pynvml is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES contains GPU UUIDs, but pynvml is unavailable to map them "
            f"for vLLM: {unresolved}"
        )

    try:
        pynvml.nvmlInit()
        uuid_to_index: Dict[str, int] = {}
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            uuid_to_index[str(uuid)] = idx

        resolved: List[str] = []
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


def maybe_patch_cuda_visible_devices_for_vllm() -> None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        return
    normalized = normalize_visible_devices_csv(value)
    if normalized and normalized != value:
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized
        print(f"[vllm] normalized CUDA_VISIBLE_DEVICES for vLLM: {value} -> {normalized}", flush=True)


def configure_vllm_environment() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    maybe_patch_cuda_visible_devices_for_vllm()


def batched(items: List[Tuple[int, Dict]], batch_size: int) -> Iterable[List[Tuple[int, Dict]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def get_dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def init_distributed(local_rank: int, world_size: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def visible_device_indices() -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        indices = []
        normalized = normalize_visible_devices_csv(visible)
        for token in normalized.split(","):
            token = token.strip()
            if token.isdigit():
                indices.append(int(token))
        if indices:
            return indices
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def emit_autotune_memory_stats(rank: int) -> None:
    if rank != 0 or pynvml is None or not torch.cuda.is_available():
        return

    device_indices = visible_device_indices()
    if not device_indices:
        return

    try:
        pynvml.nvmlInit()
        best_payload = None
        for device_index in device_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = int(info.total)
            used_memory = int(info.used)
            if total_memory <= 0:
                continue
            utilization = used_memory / total_memory
            payload = {
                "phase": "reference",
                "device_index": int(device_index),
                "peak_allocated_bytes": used_memory,
                "peak_reserved_bytes": used_memory,
                "current_allocated_bytes": used_memory,
                "current_reserved_bytes": used_memory,
                "total_memory_bytes": total_memory,
                "peak_allocated_utilization": round(utilization, 6),
                "peak_reserved_utilization": round(utilization, 6),
                "current_reserved_utilization": round(utilization, 6),
                "memory_stats_kind": "nvml_current",
            }
            if best_payload is None or payload["peak_reserved_utilization"] > best_payload["peak_reserved_utilization"]:
                best_payload = payload
        if best_payload is not None:
            print(f"[autobatch-memory] {json.dumps(best_payload, sort_keys=True)}", flush=True)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def init_generation_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def init_generation_engine(model_name: str, model_path: str) -> Tuple[AutoTokenizer, torch.nn.Module]:
    attn_impl = "eager" if model_name.startswith("gemma2-") else "sdpa"
    tokenizer = init_generation_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=_default_dtype(),
        attn_implementation=attn_impl,
    )
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return tokenizer, model


def init_reward_engine(model_name: str, reward_model_path: str, device: torch.device | None = None) -> torch.nn.Module:
    attn_impl = "eager" if model_name.startswith("gemma2-") else "sdpa"
    model_cls = (
        AutoModelForCausalLMWithScalarHeadODIN
        if (Path(reward_model_path) / "scalar_head_quality.pt").exists()
        else AutoModelForCausalLMWithScalarHead
    )
    model = model_cls.from_pretrained(
        reward_model_path,
        trust_remote_code=True,
        torch_dtype=_default_dtype(),
        attn_implementation=attn_impl,
    )
    if device is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def load_and_store_canonical_rows(dataset_name: str, split: str) -> List[Dict]:
    raw_path = dataset_split_path(dataset_name, split)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)
    canonical_rows = [canonicalize_preference_row(dataset_name, row) for row in raw_rows]

    out_path = Path(canonical_split_path(dataset_name, split))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(canonical_rows, f, ensure_ascii=False, indent=2)
    return canonical_rows


def build_formatted_prompts(batch_rows: List[Tuple[int, Dict]], dataset_name: str) -> List[str]:
    return [
        format_canonical_prompt(dataset_name, row["prompt"], "<|user|>", "<|assistant|>")
        for _, row in batch_rows
    ]


def build_token_budgeted_prompt_chunks(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    max_batch_prompts: int,
    max_batch_tokens: int,
    output_tokens_per_prompt: int,
) -> List[Tuple[List[int], List[str]]]:
    chunks: List[Tuple[List[int], List[str]]] = []
    current_indices: List[int] = []
    current_prompts: List[str] = []
    current_tokens = 0

    for idx, prompt in enumerate(prompts):
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False)) + max(output_tokens_per_prompt, 0)
        token_count = max(token_count, 1)
        overflow_prompts = len(current_prompts) >= max_batch_prompts
        overflow_tokens = current_prompts and (current_tokens + token_count > max_batch_tokens)
        if overflow_prompts or overflow_tokens:
            chunks.append((current_indices, current_prompts))
            current_indices = []
            current_prompts = []
            current_tokens = 0
        current_indices.append(idx)
        current_prompts.append(prompt)
        current_tokens += token_count

    if current_prompts:
        chunks.append((current_indices, current_prompts))
    return chunks


def effective_prompt_batch_size(args) -> int:
    if args.generation_backend == "vllm":
        return max(1, args.batch_size_per_gpu, args.vllm_max_batch_prompts)
    return max(1, args.batch_size_per_gpu)


def init_vllm_generation_engine(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    dtype: str,
):
    configure_vllm_environment()
    from vllm import LLM

    print(
        "[vllm] initializing reference engine "
        f"model={model_path} tp={tensor_parallel_size} dtype={dtype} "
        f"gpu_memory_utilization={gpu_memory_utilization} max_model_len={max_model_len} "
        f"enforce_eager={enforce_eager} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', '')}",
        flush=True,
    )
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )


def generate_responses_for_batch(
    formatted_prompts: List[str],
    tokenizer: AutoTokenizer,
    generation_model: torch.nn.Module,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> List[List[str]]:
    device = next(generation_model.parameters()).device
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = generation_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=num_return_sequences,
        )
    decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    grouped = [[] for _ in formatted_prompts]
    for idx, text in enumerate(decoded):
        grouped[idx // num_return_sequences].append(text.strip())
    return grouped


def generate_responses_for_batch_vllm(
    formatted_prompts: List[str],
    tokenizer: AutoTokenizer,
    generation_engine,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
    max_batch_prompts: int,
    max_batch_tokens: int,
) -> List[List[str]]:
    from vllm import SamplingParams

    grouped = [[] for _ in formatted_prompts]
    sampling_params = SamplingParams(
        n=num_return_sequences,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        skip_special_tokens=True,
    )
    for indices, prompt_chunk in build_token_budgeted_prompt_chunks(
        formatted_prompts,
        tokenizer,
        max_batch_prompts=max_batch_prompts,
        max_batch_tokens=max_batch_tokens,
        output_tokens_per_prompt=max_new_tokens * max(1, num_return_sequences),
    ):
        outputs = generation_engine.generate(prompt_chunk, sampling_params, use_tqdm=False)
        for original_idx, request_output in zip(indices, outputs):
            grouped[original_idx].extend(output.text.strip() for output in request_output.outputs)
    return grouped


def release_generation_engine(generation_engine) -> None:
    if generation_engine is None:
        return
    del generation_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenize_prompt_generation(
    tokenizer: AutoTokenizer,
    prompt: str,
    generation: str,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str = "keep_start",
) -> Dict[str, List[int]]:
    prompt_token_ids = tokenizer.encode(prompt)
    generation_token_ids = tokenizer.encode(generation)

    if prompt_token_ids and prompt_token_ids[-1] == tokenizer.eos_token_id:
        prompt_token_ids.pop()
    if generation_token_ids and tokenizer.bos_token_id is not None and generation_token_ids[0] == tokenizer.bos_token_id:
        generation_token_ids.pop(0)

    if (len(prompt_token_ids) + len(generation_token_ids) > max_length) and (len(prompt_token_ids) > max_prompt_length):
        if truncation_mode == "keep_start":
            prompt_token_ids = prompt_token_ids[:max_prompt_length]
        elif truncation_mode == "keep_end":
            prompt_token_ids = prompt_token_ids[-max_prompt_length:]
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    if len(prompt_token_ids) + len(generation_token_ids) > max_length:
        generation_token_ids = generation_token_ids[: max_length - len(prompt_token_ids)]

    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
    generation_text = tokenizer.decode(generation_token_ids, skip_special_tokens=True)
    if tokenizer.eos_token:
        generation_text = generation_text + tokenizer.eos_token

    prompt_tokens = tokenizer(prompt_text)
    combined_tokens = tokenizer(prompt_text + generation_text)
    labels = combined_tokens["input_ids"][:]
    prompt_label_len = max(len(prompt_tokens["input_ids"]) - 1, 0)
    labels[:prompt_label_len] = [-100] * prompt_label_len

    return {
        "input_ids": combined_tokens["input_ids"],
        "attention_mask": combined_tokens["attention_mask"],
        "labels": labels,
    }


def collate_reward_batch(tokenizer: AutoTokenizer, encodings: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    padded = {}
    input_tensors = [torch.LongTensor(item["input_ids"]) for item in encodings]
    mask_tensors = [torch.LongTensor(item["attention_mask"]) for item in encodings]
    label_tensors = [torch.LongTensor(item["labels"]) for item in encodings]

    padded["input_ids"] = pad_sequence(input_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded["attention_mask"] = pad_sequence(mask_tensors, batch_first=True, padding_value=0)
    padded["labels"] = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    return padded


def score_responses_for_batch(
    formatted_prompts: List[str],
    responses_by_prompt: List[List[str]],
    tokenizer: AutoTokenizer,
    reward_model: torch.nn.Module,
    max_length: int,
    max_prompt_length: int,
    gen_valid_len: int,
    penalty_per_token: float,
    reward_batch_size_per_gpu: int | None = None,
) -> List[List[float]]:
    device = next(reward_model.parameters()).device
    flat_encodings = []
    counts = []
    for prompt, responses in zip(formatted_prompts, responses_by_prompt):
        counts.append(len(responses))
        for response in responses:
            flat_encodings.append(
                tokenize_prompt_generation(
                    tokenizer,
                    prompt,
                    response,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                )
            )

    if not flat_encodings:
        return [[] for _ in formatted_prompts]

    if reward_batch_size_per_gpu is None or reward_batch_size_per_gpu <= 0:
        reward_batch_size_per_gpu = len(flat_encodings)

    flat_rewards = []
    for start in range(0, len(flat_encodings), reward_batch_size_per_gpu):
        chunk = flat_encodings[start : start + reward_batch_size_per_gpu]
        batch = collate_reward_batch(tokenizer, chunk)
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.inference_mode():
            rewards = reward_model(batch["input_ids"], attention_mask=batch["attention_mask"])
            if isinstance(rewards, tuple):
                rewards = rewards[0]
            masks = batch["labels"] != -100
            last_token_indices = masks.long().sum(dim=1) - 1
            gather_indices = last_token_indices.to(device=device, dtype=torch.long).unsqueeze(1)
            last_token_rewards = torch.gather(rewards, dim=1, index=gather_indices).squeeze(-1)
            if gen_valid_len > 0 and penalty_per_token > 0.0:
                response_lengths = masks.detach().float().sum(dim=-1)
                excess_lengths = (response_lengths - gen_valid_len).clamp_min(0)
                penalties = excess_lengths.to(device=device, dtype=last_token_rewards.dtype) * penalty_per_token
                last_token_rewards = last_token_rewards - penalties

        flat_rewards.extend(float(item) for item in last_token_rewards.detach().cpu().tolist())
    grouped_rewards = []
    start = 0
    for count in counts:
        grouped_rewards.append(flat_rewards[start : start + count])
        start += count
    return grouped_rewards


def generate_and_score_batch(
    batch_rows: List[Tuple[int, Dict]],
    args,
    tokenizer: AutoTokenizer,
    generation_model,
    generation_engine,
    reward_model: torch.nn.Module,
) -> Dict[int, Dict]:
    formatted_prompts = build_formatted_prompts(batch_rows, args.dataset)
    responses_by_prompt = [[] for _ in batch_rows]
    dataset_preset = DATASET_PRESETS[args.dataset]

    refs_per_generation_call = max(1, min(args.refs_per_generation_call, args.num_refs))
    for start_ref in range(0, args.num_refs, refs_per_generation_call):
        current_num_return_sequences = min(refs_per_generation_call, args.num_refs - start_ref)
        if args.generation_backend == "vllm":
            grouped_responses = generate_responses_for_batch_vllm(
                formatted_prompts,
                tokenizer,
                generation_engine,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                current_num_return_sequences,
                args.vllm_max_batch_prompts,
                args.vllm_max_batch_tokens,
            )
        else:
            grouped_responses = generate_responses_for_batch(
                formatted_prompts,
                tokenizer,
                generation_model,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                current_num_return_sequences,
            )
        for idx, response_group in enumerate(grouped_responses):
            responses_by_prompt[idx].extend(response_group)

    rewards_by_prompt = score_responses_for_batch(
        formatted_prompts,
        responses_by_prompt,
        tokenizer,
        reward_model,
        args.max_length,
        args.max_prompt_length,
        dataset_preset.gen_valid_len,
        dataset_preset.penalty_per_token,
        args.reward_batch_size_per_gpu,
    )

    return {
        row_index: {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "ref_responses": responses,
            "ref_rewards": rewards,
        }
        for (row_index, row), responses, rewards in zip(batch_rows, responses_by_prompt, rewards_by_prompt)
    }


def run_autotune_probe(
    args,
    rank: int,
    world_size: int,
    tokenizer: AutoTokenizer,
    generation_model,
    generation_engine,
    reward_model: torch.nn.Module,
) -> None:
    split = "train_prefs"
    if rank == 0:
        canonical_rows = load_and_store_canonical_rows(args.dataset, split)
        print(
            f"[autotune] prepared canonical split {canonical_split_path(args.dataset, split)} "
            f"with {len(canonical_rows)} rows"
        )
    barrier(world_size)

    with open(canonical_split_path(args.dataset, split), "r", encoding="utf-8") as f:
        rows = json.load(f)

    indexed_rows = list(enumerate(rows))
    shard_rows = indexed_rows[rank::world_size]
    probe_batch_rows = shard_rows[: effective_prompt_batch_size(args)]
    if not probe_batch_rows:
        raise RuntimeError("Autotune probe could not find any rows in the reference dataset shard.")

    _ = generate_and_score_batch(
        probe_batch_rows,
        args,
        tokenizer,
        generation_model,
        generation_engine,
        reward_model,
    )
    barrier(world_size)
    emit_autotune_memory_stats(rank)
    if rank == 0:
        print(
            "[autotune] reference generation probe succeeded "
            f"batch_size_per_gpu={args.batch_size_per_gpu} "
            f"reward_batch_size_per_gpu={args.reward_batch_size_per_gpu}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reference responses and reward scores for train_prefs/test_prefs."
    )
    parser.add_argument("--dataset", required=True, choices=["ultrafb_bin", "hh_rlhf"])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True, help="Path to the SFT checkpoint, typically .../final_hf")
    parser.add_argument("--reward_model_path", required=True, help="Path to the reward checkpoint, typically .../final_hf")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_refs", type=int, default=5)
    parser.add_argument("--refs_per_generation_call", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1212)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--reward_batch_size_per_gpu", type=int, default=None)
    parser.add_argument("--generation_backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_max_batch_prompts", type=int, default=None)
    parser.add_argument("--vllm_max_batch_tokens", type=int, default=None)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--reward_device_index", type=int, default=None)
    parser.add_argument("--autotune_probe_only", action="store_true")
    args = parser.parse_args()

    if args.vllm_max_model_len is None:
        args.vllm_max_model_len = max(args.max_length, args.max_prompt_length + args.max_new_tokens)
    if args.vllm_max_batch_prompts is None:
        args.vllm_max_batch_prompts = args.batch_size_per_gpu
    if args.vllm_max_batch_tokens is None:
        args.vllm_max_batch_tokens = args.vllm_max_batch_prompts * args.vllm_max_model_len

    rank, local_rank, world_size = get_dist_info()
    if args.generation_backend == "vllm" and world_size > 1:
        raise ValueError("generation_backend=vllm should be launched as a single process; do not wrap it with torchrun.")
    if args.generation_backend != "vllm":
        init_distributed(local_rank, world_size)

    dataset_dir = dataset_dir_name(args.dataset)
    shard_root = PROJECT_ROOT / "data" / dataset_dir / ".reference_response_shards" / args.model_name
    if rank == 0 and shard_root.exists():
        shutil.rmtree(shard_root)
    barrier(world_size)
    shard_root.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    if rank == 0:
        print(
            f"[generate_reference_responses] dataset={args.dataset} model_name={args.model_name} "
            f"world_size={world_size} num_refs={args.num_refs} "
            f"refs_per_generation_call={args.refs_per_generation_call} "
            f"generation_backend={args.generation_backend}"
        )

    tokenizer = init_generation_tokenizer(args.model_path)
    generation_model = None
    generation_engine = None
    if args.generation_backend == "hf":
        _, generation_model = init_generation_engine(args.model_name, args.model_path)
    else:
        generation_engine = init_vllm_generation_engine(
            args.model_path,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
            dtype=args.vllm_dtype,
        )

    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        reward_device_index = args.reward_device_index
        if reward_device_index is None:
            if args.generation_backend == "vllm" and visible_gpu_count > args.vllm_tensor_parallel_size:
                # vLLM occupies [0, vllm_tp), put reward model on the first free GPU.
                reward_device_index = args.vllm_tensor_parallel_size
            else:
                # torchrun (hf backend): each rank must use its own GPU, not GPU 0.
                reward_device_index = local_rank
        reward_device_index = min(max(reward_device_index, 0), max(visible_gpu_count - 1, 0))
        reward_device = torch.device("cuda", reward_device_index)
    else:
        reward_device = torch.device("cpu")
    if rank == 0:
        print(f"[reward] initializing reward scorer on device={reward_device}", flush=True)

    reward_model = init_reward_engine(args.model_name, args.reward_model_path, device=reward_device)

    if args.autotune_probe_only:
        run_autotune_probe(
            args,
            rank,
            world_size,
            tokenizer,
            generation_model,
            generation_engine,
            reward_model,
        )
        release_generation_engine(generation_model)
        release_generation_engine(generation_engine)
        del reward_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rank == 0 and shard_root.exists():
            shutil.rmtree(shard_root)
        cleanup_distributed(world_size)
        return

    for split in ["train_prefs", "test_prefs"]:
        if rank == 0:
            canonical_rows = load_and_store_canonical_rows(args.dataset, split)
            print(f"Wrote canonical split to {canonical_split_path(args.dataset, split)} with {len(canonical_rows)} rows")
        barrier(world_size)

        with open(canonical_split_path(args.dataset, split), "r", encoding="utf-8") as f:
            rows = json.load(f)

        indexed_rows = list(enumerate(rows))
        shard_rows = indexed_rows[rank::world_size]
        shard_path = shard_root / f"{split}_rank_{rank:05d}.json"

        shard_payload = []
        prompt_batch_size = effective_prompt_batch_size(args)
        batch_iterator = batched(shard_rows, prompt_batch_size)
        total_batches = (len(shard_rows) + prompt_batch_size - 1) // prompt_batch_size
        if rank == 0:
            batch_iterator = tqdm.tqdm(
                batch_iterator,
                total=total_batches,
                desc=f"Generating {split} refs",
            )
        for batch_rows in batch_iterator:
            batch_payload = generate_and_score_batch(
                batch_rows,
                args,
                tokenizer,
                generation_model,
                generation_engine,
                reward_model,
            )

            shard_payload.extend(
                {"index": row_index, **payload}
                for row_index, payload in sorted(batch_payload.items(), key=lambda item: item[0])
            )

        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(shard_payload, f, ensure_ascii=False, indent=2)

        barrier(world_size)

        if rank == 0:
            merged = []
            for shard_file in sorted(shard_root.glob(f"{split}_rank_*.json")):
                with open(shard_file, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))
            merged.sort(key=lambda item: item["index"])
            final_rows = [
                {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "ref_responses": item["ref_responses"],
                    "ref_rewards": item["ref_rewards"],
                }
                for item in merged
            ]
            out_path = PROJECT_ROOT / "data" / dataset_dir / f"{args.model_name}_{split}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_rows, f, ensure_ascii=False, indent=2)
            print(f"Wrote enriched dataset to {out_path} with {len(final_rows)} rows")

        barrier(world_size)

    if rank == 0 and shard_root.exists():
        shutil.rmtree(shard_root)

    release_generation_engine(generation_model)
    release_generation_engine(generation_engine)
    del reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()

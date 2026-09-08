from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


OOM_PATTERNS = (
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "hip out of memory",
    "outofmemoryerror",
    "torch.cuda.outofmemoryerror",
    "cuda error: out of memory",
    "cuda error: an illegal memory access",
    "nccl watchdog thread terminated",
    "nccl error",
)
CAPACITY_FAILURE_PATTERNS = (
    "evaluation iterator is empty; cannot run autotune eval probe",
    "processgroupnccl.cpp",
    "signal 6 (sigabrt)",
    "signal 9 (sigkill)",
    "signal 11 (sigsegv)",
)
MEMORY_MARKER_PREFIX = "[autobatch-memory] "


@dataclass
class TrialResult:
    batch_size: int
    returncode: int
    succeeded: bool
    oom: bool
    capacity_failure: bool
    elapsed_seconds: float
    output: str
    memory_stats: dict | None = None


def sanitize_label(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    return sanitized.strip("_") or "batch"


def build_cache_path(
    cache_root: Path,
    label: str,
    shell_command: str,
    min_memory_utilization: float,
    target_memory_utilization: float,
    max_memory_utilization: float,
) -> Path:
    payload = {
        "label": label,
        "shell_command": shell_command,
        "min_memory_utilization": float(min_memory_utilization),
        "target_memory_utilization": float(target_memory_utilization),
        "max_memory_utilization": float(max_memory_utilization),
        "python": sys.executable,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "cwd": os.getcwd(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{sanitize_label(label)}-{digest}.json"


def build_label_cache_path(cache_root: Path, label: str) -> Path:
    return cache_root / f"{sanitize_label(label)}-latest.json"


def is_oom_output(output: str) -> bool:
    lowered = (output or "").lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


def is_capacity_failure_output(output: str, returncode: int) -> bool:
    lowered = (output or "").lower()
    if any(pattern in lowered for pattern in CAPACITY_FAILURE_PATTERNS):
        return True
    return returncode < 0 and "processgroupnccl" in lowered


def tail_output(output: str, max_chars: int = 20000) -> str:
    output = output or ""
    if len(output) <= max_chars:
        return output
    return output[-max_chars:]


def extract_memory_stats(output: str) -> dict | None:
    for line in reversed((output or "").splitlines()):
        if not line.startswith(MEMORY_MARKER_PREFIX):
            continue
        try:
            return json.loads(line[len(MEMORY_MARKER_PREFIX) :].strip())
        except json.JSONDecodeError:
            return None
    return None


def load_cached_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def latest_summary_matches_memory_policy(summary: dict, args: argparse.Namespace) -> bool:
    for key, current in (
        ("min_memory_utilization", args.min_memory_utilization),
        ("target_memory_utilization", args.target_memory_utilization),
        ("max_memory_utilization", args.max_memory_utilization),
    ):
        try:
            cached = float(summary[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isclose(cached, float(current), rel_tol=0.0, abs_tol=1e-9):
            return False
    return True


def clamp_batch_size(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def next_growth_probe(current: int, maximum: int, growth_factor: float) -> int:
    grown = int(math.ceil(current * growth_factor))
    return clamp_batch_size(max(current + 1, grown), current + 1, maximum)


def biased_refinement_probe(low: int, high: int, bias: float) -> int:
    if high <= low:
        return low
    span = high - low
    candidate = low + max(1, int(math.floor(span * bias)))
    return clamp_batch_size(candidate, low + 1, high)


def trial_memory_utilization(result: TrialResult) -> float | None:
    if not result.memory_stats:
        return None
    utilization = result.memory_stats.get("peak_reserved_utilization")
    if utilization is None:
        peak_reserved = result.memory_stats.get("peak_reserved_bytes")
        total_memory = result.memory_stats.get("total_memory_bytes")
        if peak_reserved is None or total_memory in (None, 0):
            return None
        utilization = float(peak_reserved) / float(total_memory)
    return float(utilization)


def estimate_batch_from_utilization(
    batch_size: int,
    utilization: float,
    target_utilization: float,
    minimum: int,
    maximum: int,
    cap_growth_factor: float,
    safety_factor: float,
) -> int:
    utilization = max(float(utilization), 1e-6)
    target_utilization = max(float(target_utilization), utilization)
    estimated = int(math.floor(batch_size * (target_utilization / utilization) * safety_factor))
    if estimated <= batch_size:
        estimated = batch_size + 1
    cap = max(batch_size + 1, int(math.ceil(batch_size * cap_growth_factor)))
    estimated = min(estimated, cap)
    return clamp_batch_size(estimated, minimum, maximum)


def run_trial(shell: str, shell_command: str, batch_size: int, timeout_seconds: int) -> TrialResult:
    env = os.environ.copy()
    env["BATCH_SIZE"] = str(batch_size)
    env["AUTOTUNE_BATCH_SIZE_TRIAL"] = "1"

    started = time.time()
    try:
        completed = subprocess.run(
            shell_command,
            shell=True,
            executable=shell,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        output = completed.stdout or ""
        oom = is_oom_output(output)
        capacity_failure = is_capacity_failure_output(output, completed.returncode)
        succeeded = completed.returncode == 0
        memory_stats = extract_memory_stats(output)
        return TrialResult(
            batch_size=batch_size,
            returncode=completed.returncode,
            succeeded=succeeded,
            oom=oom,
            capacity_failure=capacity_failure,
            elapsed_seconds=time.time() - started,
            output=output,
            memory_stats=memory_stats,
        )
    except subprocess.TimeoutExpired as exc:
        output = ""
        if exc.stdout:
            output += exc.stdout
        if exc.stderr:
            output += exc.stderr
        oom = is_oom_output(output)
        capacity_failure = is_capacity_failure_output(output, 124)
        return TrialResult(
            batch_size=batch_size,
            returncode=124,
            succeeded=False,
            oom=oom,
            capacity_failure=capacity_failure,
            elapsed_seconds=time.time() - started,
            output=output,
            memory_stats=extract_memory_stats(output),
        )


def ensure_success_or_oom(result: TrialResult) -> None:
    if result.succeeded or result.oom or result.capacity_failure:
        return
    raise RuntimeError(
        "Autotune trial failed for a non-OOM reason.\n"
        f"batch_size={result.batch_size} returncode={result.returncode}\n"
        f"Command output tail:\n{tail_output(result.output)}"
    )


def search_best_batch(args, initial_batch: int) -> tuple[int, dict[int, TrialResult]]:
    tried: dict[int, TrialResult] = {}

    def evaluate(candidate: int) -> TrialResult:
        candidate = max(args.min_batch_size, min(args.max_batch_size, int(candidate)))
        if candidate in tried:
            return tried[candidate]
        result = run_trial(args.shell, args.shell_command, candidate, args.timeout_seconds)
        utilization = trial_memory_utilization(result)
        if result.succeeded and utilization is not None and utilization > args.max_memory_utilization:
            result = TrialResult(
                batch_size=result.batch_size,
                returncode=result.returncode,
                succeeded=False,
                oom=False,
                capacity_failure=True,
                elapsed_seconds=result.elapsed_seconds,
                output=(
                    result.output
                    + f"\n[autobatch] memory upper bound exceeded: util={utilization:.6f} "
                    + f"max={args.max_memory_utilization:.6f}\n"
                ),
                memory_stats=result.memory_stats,
            )
        tried[candidate] = result
        status = "ok"
        if not result.succeeded:
            if result.oom:
                status = "oom"
            elif result.capacity_failure:
                status = "capacity"
            else:
                status = f"rc={result.returncode}"
        utilization_text = f" util={utilization:.3f}" if utilization is not None else ""
        print(
            f"[autobatch] label={args.label} batch_size={candidate} status={status} "
            f"elapsed={result.elapsed_seconds:.2f}s{utilization_text}",
            file=sys.stderr,
            flush=True,
        )
        ensure_success_or_oom(result)
        return result

    def fast_search(initial: int) -> int:
        initial_result = evaluate(initial)
        if initial_result.succeeded:
            safe_batch = initial
            safe_result = initial_result
            utilization = trial_memory_utilization(safe_result)
            if utilization is not None:
                for _ in range(max(args.max_memory_guided_trials, 0)):
                    if utilization >= args.min_memory_utilization:
                        return safe_batch
                    target_probe = estimate_batch_from_utilization(
                        safe_batch,
                        utilization,
                        args.target_memory_utilization,
                        args.min_batch_size,
                        args.max_batch_size,
                        args.target_growth_cap,
                        args.target_safety_factor,
                    )
                    if target_probe <= safe_batch:
                        return safe_batch

                    target_result = evaluate(target_probe)
                    if target_result.succeeded:
                        safe_batch = target_probe
                        safe_result = target_result
                        utilization = trial_memory_utilization(safe_result)
                        if utilization is None:
                            break
                        continue

                    high = target_probe - 1
                    if high <= safe_batch:
                        return safe_batch

                    refine_probe = biased_refinement_probe(safe_batch, high, args.target_refine_bias)
                    if refine_probe <= safe_batch:
                        return safe_batch

                    refine_result = evaluate(refine_probe)
                    if refine_result.succeeded:
                        safe_batch = refine_probe
                        safe_result = refine_result
                        utilization = trial_memory_utilization(safe_result)
                    return safe_batch
                return safe_batch

            if safe_batch >= args.max_batch_size:
                return safe_batch

            upper_probe = next_growth_probe(safe_batch, args.max_batch_size, args.growth_factor)
            if upper_probe == safe_batch:
                return safe_batch

            upper_result = evaluate(upper_probe)
            if upper_result.succeeded:
                return upper_probe

            high = upper_probe - 1
            if high <= safe_batch:
                return safe_batch

            refine_probe = biased_refinement_probe(safe_batch, high, args.fast_refine_bias)
            if refine_probe <= safe_batch:
                return safe_batch

            refine_result = evaluate(refine_probe)
            if refine_result.succeeded:
                return refine_probe
            return safe_batch

        lower_probe = clamp_batch_size(
            int(math.floor(initial / max(args.growth_factor, 1.01))),
            args.min_batch_size,
            max(args.min_batch_size, initial - 1),
        )
        if lower_probe == initial and initial > args.min_batch_size:
            lower_probe = initial - 1
        if lower_probe < initial:
            lower_result = evaluate(lower_probe)
            if lower_result.succeeded:
                return lower_probe

        if lower_probe != args.min_batch_size:
            floor_result = evaluate(args.min_batch_size)
            if floor_result.succeeded:
                return args.min_batch_size

        raise RuntimeError(
            "Autotune could not find a safe batch size in fast mode.\n"
            f"initial_batch_size={initial}\n"
            f"minimum_batch_size={args.min_batch_size}\n"
            f"Command output tail:\n{tail_output(tried[min(tried.keys())].output if tried else '')}"
        )

    safe_batch = None
    initial = max(args.min_batch_size, min(initial_batch, args.max_batch_size))
    if args.search_mode == "fast":
        return fast_search(initial), tried

    initial_result = evaluate(initial)
    if initial_result.succeeded:
        safe_batch = initial
        probe = initial
        while probe < args.max_batch_size:
            next_probe = next_growth_probe(probe, args.max_batch_size, args.growth_factor)
            if next_probe == probe:
                break
            result = evaluate(next_probe)
            if result.succeeded:
                safe_batch = next_probe
                probe = next_probe
            else:
                break
        low = safe_batch
        high = args.max_batch_size
        for batch_size, result in tried.items():
            if batch_size > low and not result.succeeded:
                high = min(high, batch_size - 1)
        while low < high:
            mid = (low + high + 1) // 2
            result = evaluate(mid)
            if result.succeeded:
                low = mid
                safe_batch = mid
            else:
                high = mid - 1
        return safe_batch, tried

    floor_result = evaluate(args.min_batch_size)
    if not floor_result.succeeded:
        raise RuntimeError(
            "Autotune could not find a safe batch size; even the minimum batch failed.\n"
            f"minimum_batch_size={args.min_batch_size}\n"
            f"Command output tail:\n{tail_output(floor_result.output)}"
        )

    low = args.min_batch_size
    high = initial - 1
    safe_batch = low
    while low < high:
        mid = (low + high + 1) // 2
        result = evaluate(mid)
        if result.succeeded:
            low = mid
            safe_batch = mid
        else:
            high = mid - 1
    return safe_batch, tried


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe and cache the largest safe per-GPU batch size.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--shell-command", required=True, help="Shell command that uses the BATCH_SIZE env var.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--base-batch-size", type=int, required=True)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--shell", default="/bin/bash")
    parser.add_argument("--search-mode", choices=["fast", "precise"], default="fast")
    parser.add_argument("--growth-factor", type=float, default=None)
    parser.add_argument("--fast-refine-bias", type=float, default=0.4)
    parser.add_argument("--min-memory-utilization", type=float, default=0.70)
    parser.add_argument("--target-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-memory-utilization", type=float, default=0.90)
    parser.add_argument("--memory-tolerance", type=float, default=0.02)
    parser.add_argument("--target-growth-cap", type=float, default=3.0)
    parser.add_argument("--target-safety-factor", type=float, default=0.98)
    parser.add_argument("--target-refine-bias", type=float, default=0.75)
    parser.add_argument("--max-memory-guided-trials", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_batch_size = max(int(args.base_batch_size), 1)
    args.min_batch_size = max(int(args.min_batch_size), 1)
    args.max_batch_size = max(int(args.max_batch_size), args.min_batch_size)
    args.timeout_seconds = max(int(args.timeout_seconds), 60)
    if args.growth_factor is None:
        args.growth_factor = 1.25 if args.search_mode == "fast" else 2.0
    args.growth_factor = max(float(args.growth_factor), 1.05)
    args.fast_refine_bias = min(max(float(args.fast_refine_bias), 0.05), 0.95)
    args.min_memory_utilization = min(max(float(args.min_memory_utilization), 0.0), 0.995)
    args.max_memory_utilization = min(max(float(args.max_memory_utilization), 0.5), 0.995)
    if args.max_memory_utilization < args.min_memory_utilization:
        args.max_memory_utilization = args.min_memory_utilization
    args.target_memory_utilization = min(max(float(args.target_memory_utilization), args.min_memory_utilization), args.max_memory_utilization)
    args.memory_tolerance = min(max(float(args.memory_tolerance), 0.0), 0.2)
    args.target_growth_cap = max(float(args.target_growth_cap), 1.1)
    args.target_safety_factor = min(max(float(args.target_safety_factor), 0.7), 1.0)
    args.target_refine_bias = min(max(float(args.target_refine_bias), 0.05), 0.95)
    args.max_memory_guided_trials = max(int(args.max_memory_guided_trials), 0)

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = build_cache_path(
        cache_root,
        args.label,
        args.shell_command,
        args.min_memory_utilization,
        args.target_memory_utilization,
        args.max_memory_utilization,
    )
    label_cache_path = build_label_cache_path(cache_root, args.label)

    if cache_path.exists() and not args.force:
        cached = json.loads(cache_path.read_text())
        print(f"[autobatch] cache_hit label={args.label} batch_size={cached['batch_size']}", file=sys.stderr, flush=True)
        print(int(cached["batch_size"]))
        return

    initial_batch = args.base_batch_size
    latest_cached = load_cached_summary(label_cache_path)
    if (
        latest_cached is not None
        and "batch_size" in latest_cached
        and latest_summary_matches_memory_policy(latest_cached, args)
    ):
        initial_batch = clamp_batch_size(int(latest_cached["batch_size"]), args.min_batch_size, args.max_batch_size)
        print(
            f"[autobatch] seed_hit label={args.label} batch_size={initial_batch} path={label_cache_path}",
            file=sys.stderr,
            flush=True,
        )
    elif latest_cached is not None and "batch_size" in latest_cached:
        print(
            f"[autobatch] seed_skip label={args.label} path={label_cache_path} reason=memory_policy_changed",
            file=sys.stderr,
            flush=True,
        )

    best_batch, tried = search_best_batch(args, initial_batch)
    summary = {
        "label": args.label,
        "batch_size": int(best_batch),
        "base_batch_size": int(args.base_batch_size),
        "initial_batch_size": int(initial_batch),
        "min_batch_size": int(args.min_batch_size),
        "max_batch_size": int(args.max_batch_size),
        "search_mode": args.search_mode,
        "growth_factor": args.growth_factor,
        "min_memory_utilization": args.min_memory_utilization,
        "target_memory_utilization": args.target_memory_utilization,
        "max_memory_utilization": args.max_memory_utilization,
        "shell_command": args.shell_command,
        "trials": [
            {
                "batch_size": int(batch_size),
                "returncode": result.returncode,
                "succeeded": result.succeeded,
                "oom": result.oom,
                "capacity_failure": result.capacity_failure,
                "elapsed_seconds": round(result.elapsed_seconds, 4),
                "memory_utilization": (
                    round(trial_memory_utilization(result), 6)
                    if trial_memory_utilization(result) is not None
                    else None
                ),
                "memory_stats": result.memory_stats,
            }
            for batch_size, result in sorted(tried.items())
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    label_cache_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[autobatch] cached label={args.label} batch_size={best_batch} path={cache_path}", file=sys.stderr, flush=True)
    print(best_batch)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from llm_comparator import LLMComparator
except ImportError:
    from .llm_comparator import LLMComparator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_completed_log() -> Path:
    pipeline_state_root = os.getenv("PIPELINE_STATE_ROOT")
    root = Path(pipeline_state_root) if pipeline_state_root else _project_root() / "pipeline_state"
    return root / "05_rating" / "completed.log"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def collapse_text(text: str, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def extract_step_number(step_name: str) -> int:
    try:
        return int(step_name.split("_")[-1])
    except Exception:
        return 0


def load_completed_labels(completed_log: Path) -> set[str]:
    if not completed_log.exists():
        return set()
    with completed_log.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_completed_label(completed_log: Path, label: str) -> None:
    completed_log.parent.mkdir(parents=True, exist_ok=True)
    existing = load_completed_labels(completed_log)
    if label in existing:
        return
    with completed_log.open("a", encoding="utf-8") as handle:
        handle.write(label + "\n")


def step_pipeline_label(run_name: str, step_name: str) -> str:
    return f"rating:{run_name}:{step_name}"


def step_dirs_for_run(run_dir: Path) -> list[Path]:
    sample_dir = run_dir / "sample_on_test"
    if not sample_dir.exists():
        return []
    step_dirs = [p for p in sample_dir.iterdir() if p.is_dir() and p.name.startswith("step_")]
    step_dirs.sort(key=lambda p: extract_step_number(p.name))
    return step_dirs


def merged_json_path(step_dir: Path) -> Path:
    return step_dir / "merged.json"


def merged_rated_json_path(step_dir: Path) -> Path:
    return step_dir / "merged_rated.json"


def rating_summary_path(run_dir: Path) -> Path:
    return run_dir / "llm_rating_summary.json"


def rating_plot_path(run_dir: Path) -> Path:
    return run_dir / "llm_rating_curve.png"


def map_final_label_to_preferred(final_label: str | None) -> tuple[str | None, float | None]:
    if final_label == "Policy":
        return "policy_response", 1.0
    if final_label == "Reference":
        return "reference_response", 0.0
    if final_label == "tie":
        return "tie", 0.5
    return None, None


def rate_step_payload(
    run_dir: str,
    step_dir: str,
    completed_log: str,
    step_progress_interval: int,
) -> dict[str, Any]:
    run_dir_path = Path(run_dir)
    step_dir_path = Path(step_dir)
    completed_log_path = Path(completed_log)
    run_name = run_dir_path.name
    step_name = step_dir_path.name
    merged_path = merged_json_path(step_dir_path)
    rated_path = merged_rated_json_path(step_dir_path)

    rows = load_json(merged_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {merged_path}")

    comparator = LLMComparator()

    total_examples = len(rows)
    rated_rows: list[dict[str, Any]] = []
    valid_examples = 0
    skipped_examples = 0
    total_cost_usd = 0.0
    first_logged = False
    log_step_progress = step_name == "step_0"

    print(
        f"[STEP-START] run={run_name} step={step_name} examples={total_examples}/{total_examples}",
        flush=True,
    )

    def maybe_log_step_progress(current_index: int) -> None:
        if not log_step_progress or step_progress_interval <= 0:
            return
        if (current_index + 1) % step_progress_interval != 0:
            return
        print(
            f"[STEP-PROGRESS] run={run_name} step={step_name} "
            f"processed={current_index + 1}/{total_examples} "
            f"valid={valid_examples} skipped={skipped_examples}",
            flush=True,
        )

    for index, original_row in enumerate(rows):
        row = copy.deepcopy(original_row)
        prompt = str(row.get("prompt") or "")
        policy_response = str(row.get("policy_response") or "")
        ref_responses = row.get("ref_responses") or []

        if not isinstance(ref_responses, list) or not ref_responses:
            row["llm_rating_skipped_reason"] = "missing_reference_response"
            rated_rows.append(row)
            skipped_examples += 1
            print(
                f"[STEP-SKIP-DATUM] run={run_name} step={step_name} index={index} reason=missing_reference_response",
                flush=True,
            )
            maybe_log_step_progress(index)
            continue

        reference_response = str(ref_responses[0] or "")

        try:
            comparison = comparator.compare_responses(policy_response, reference_response, prompt)
        except Exception as exc:
            row["llm_rating_skipped_reason"] = "judge_error"
            row["llm_rating_error"] = str(exc)
            rated_rows.append(row)
            skipped_examples += 1
            print(
                f"[STEP-SKIP-DATUM] run={run_name} step={step_name} index={index} reason=judge_error",
                flush=True,
            )
            maybe_log_step_progress(index)
            continue

        row["llm_comparison"] = comparison
        row["reference_response_for_rating"] = reference_response
        row["llm_judge_outputs"] = {
            "policy_first": comparison["policy_first"].get("raw_text", ""),
            "reference_first": comparison["reference_first"].get("raw_text", ""),
        }
        total_cost_usd += float(comparison["aggregate_usage"].get("cost_usd", 0.0) or 0.0)

        if not first_logged:
            print(
                f"[JUDGE-TEXT] run={run_name} step={step_name} "
                f"policy_first='{collapse_text(row['llm_judge_outputs']['policy_first'])}' "
                f"reference_first='{collapse_text(row['llm_judge_outputs']['reference_first'])}'",
                flush=True,
            )
            first_logged = True

        first_label = comparison["policy_first"].get("label")
        second_label = comparison["reference_first"].get("label")
        preferred_response, winrate = map_final_label_to_preferred(comparison.get("final"))

        is_valid = first_label is not None and second_label is not None and preferred_response is not None
        if not is_valid or winrate is None:
            row["preferred_response"] = None
            row["winrate"] = None
            row["llm_rating_skipped_reason"] = "unparsed_label"
            rated_rows.append(row)
            skipped_examples += 1
            print(
                f"[STEP-SKIP-DATUM] run={run_name} step={step_name} index={index} reason=unparsed_label",
                flush=True,
            )
            maybe_log_step_progress(index)
            continue

        row["preferred_response"] = preferred_response
        row["winrate"] = winrate
        valid_examples += 1
        rated_rows.append(row)
        maybe_log_step_progress(index)

    dump_json(rated_path, rated_rows)
    append_completed_label(completed_log_path, step_pipeline_label(run_name, step_name))

    policy_rewards = [
        float(row.get("policy_reward_origin"))
        for row in rated_rows
        if row.get("policy_reward_origin") is not None
    ]
    valid_winrates = [float(row["winrate"]) for row in rated_rows if row.get("winrate") is not None]

    return {
        "run_name": run_name,
        "step_name": step_name,
        "step_number": extract_step_number(step_name),
        "examples": total_examples,
        "valid_examples": valid_examples,
        "skipped_examples": skipped_examples,
        "avg_proxy_reward_origin": (
            sum(policy_rewards) / len(policy_rewards) if policy_rewards else None
        ),
        "avg_win_rate": (sum(valid_winrates) / len(valid_winrates) if valid_winrates else None),
        "api_cost_usd": total_cost_usd,
        "rated_path": str(rated_path),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    step_summaries: list[dict[str, Any]] = []
    total_cost_usd = 0.0
    total_examples = 0
    total_valid_examples = 0
    total_skipped_examples = 0

    for step_dir in step_dirs_for_run(run_dir):
        rated_path = merged_rated_json_path(step_dir)
        if not rated_path.exists():
            continue

        rows = load_json(rated_path)
        if not isinstance(rows, list):
            continue

        policy_rewards = [
            float(row.get("policy_reward_origin"))
            for row in rows
            if row.get("policy_reward_origin") is not None
        ]
        valid_winrates = [float(row["winrate"]) for row in rows if row.get("winrate") is not None]
        step_cost = sum(
            float((row.get("llm_comparison") or {}).get("aggregate_usage", {}).get("cost_usd", 0.0) or 0.0)
            for row in rows
        )
        valid_examples = len(valid_winrates)
        skipped_examples = len([row for row in rows if row.get("winrate") is None])

        total_cost_usd += step_cost
        total_examples += len(rows)
        total_valid_examples += valid_examples
        total_skipped_examples += skipped_examples

        step_summaries.append(
            {
                "step_name": step_dir.name,
                "step_number": extract_step_number(step_dir.name),
                "examples": len(rows),
                "valid_examples": valid_examples,
                "skipped_examples": skipped_examples,
                "avg_proxy_reward_origin": (
                    sum(policy_rewards) / len(policy_rewards) if policy_rewards else None
                ),
                "avg_win_rate": (sum(valid_winrates) / len(valid_winrates) if valid_winrates else None),
                "api_cost_usd": step_cost,
            }
        )

    step_summaries.sort(key=lambda item: item["step_number"])

    summary = {
        "run_name": run_dir.name,
        "steps": step_summaries,
        "totals": {
            "steps": len(step_summaries),
            "examples": total_examples,
            "valid_examples": total_valid_examples,
            "skipped_examples": total_skipped_examples,
            "api_cost_usd": total_cost_usd,
        },
    }
    dump_json(rating_summary_path(run_dir), summary)
    return summary


def plot_run_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    steps = summary.get("steps", [])
    if not steps:
        return

    x = [step["step_number"] for step in steps]
    rewards = [step["avg_proxy_reward_origin"] for step in steps]
    wins = [step["avg_win_rate"] for step in steps]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(8.6, 5.0), dpi=180)
    ax2 = ax1.twinx()

    reward_line, = ax1.plot(
        x,
        rewards,
        color="#155eef",
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        label="Proxy Reward Origin",
    )
    win_line, = ax2.plot(
        x,
        wins,
        color="#f04438",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4.0,
        label="Win Rate",
    )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Proxy Reward Origin", color="#155eef")
    ax2.set_ylabel("Win Rate", color="#f04438")
    ax1.tick_params(axis="y", colors="#155eef")
    ax2.tick_params(axis="y", colors="#f04438")
    ax2.set_ylim(0.0, 1.0)
    ax1.set_title(f"LLM Rating Summary: {run_dir.name}")

    handles = [reward_line, win_line]
    labels = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(rating_plot_path(run_dir), bbox_inches="tight")
    plt.close(fig)


def refresh_run_artifacts(run_dir: Path) -> dict[str, Any]:
    summary = summarize_run(run_dir)
    plot_run_summary(run_dir, summary)
    return summary


def spawn_step_worker(
    script_path: Path,
    run_dir: Path,
    step_dir: Path,
    step_progress_interval: int,
) -> subprocess.Popen[Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--run_dir",
        str(run_dir),
        "--step_dir",
        str(step_dir),
        "--step_progress_interval",
        str(step_progress_interval),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(cmd, env=env)


def run_rate_for_run(
    run_dir: Path,
    parallel_workers: int,
    step_progress_interval: int,
    force: bool,
) -> int:
    completed_log = _default_completed_log()
    step_dirs = [step for step in step_dirs_for_run(run_dir) if merged_json_path(step).exists()]
    if not step_dirs:
        print(f"[SKIP] rating:{run_dir.name} no merged.json found under sample_on_test/step_*", flush=True)
        return 0

    completed = load_completed_labels(completed_log)
    all_labels = {step_pipeline_label(run_dir.name, step.name) for step in step_dirs}
    run_already_rated = all_labels.issubset(completed)

    if run_already_rated and not force:
        summary = refresh_run_artifacts(run_dir)
        print(
            f"[REFRESH] run={run_dir.name} rated_already=1 total_steps={summary['totals']['steps']} "
            f"api_cost_usd={summary['totals']['api_cost_usd']:.6f}",
            flush=True,
        )
        return 0

    pending_steps = [
        step for step in step_dirs
        if force or step_pipeline_label(run_dir.name, step.name) not in completed
    ]

    workers = max(1, min(parallel_workers, len(pending_steps) if pending_steps else 1))
    print(
        f"[RATE] run={run_dir.name} total_steps={len(step_dirs)} pending_steps={len(pending_steps)} "
        f"skipped_steps={len(step_dirs) - len(pending_steps)} workers={workers}",
        flush=True,
    )

    if not pending_steps:
        summary = refresh_run_artifacts(run_dir)
        print(
            f"[DONE] run={run_dir.name} rated_steps=0 total_rated_steps={summary['totals']['steps']} "
            f"api_cost_usd={summary['totals']['api_cost_usd']:.6f}",
            flush=True,
        )
        return 0

    script_path = Path(__file__).resolve()
    active: dict[str, subprocess.Popen[Any]] = {}
    pending_queue = list(pending_steps)
    finished_steps = 0
    failed = False

    with tqdm(total=len(pending_steps), desc=f"Rating {run_dir.name}", unit="step") as progress:
        while pending_queue or active:
            while pending_queue and len(active) < workers:
                step_dir = pending_queue.pop(0)
                process = spawn_step_worker(script_path, run_dir, step_dir, step_progress_interval)
                active[step_dir.name] = process
                print(
                    f"[STEP-SPAWN] run={run_dir.name} step={step_dir.name} active_workers={len(active)}/{workers}",
                    flush=True,
                )

            time.sleep(0.2)
            finished_now: list[str] = []
            for step_name, process in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                finished_now.append(step_name)
                finished_steps += 1
                progress.update(1)
                if return_code != 0:
                    failed = True
                    print(
                        f"[ERROR] run={run_dir.name} step={step_name} exit_code={return_code}",
                        flush=True,
                    )
                else:
                    print(
                        f"[PROGRESS] run={run_dir.name} completed_steps={finished_steps}/{len(pending_steps)} last={step_name}",
                        flush=True,
                    )

            for step_name in finished_now:
                active.pop(step_name, None)

    summary = refresh_run_artifacts(run_dir)
    print(
        f"[DONE] run={run_dir.name} rated_steps={len(pending_steps)} total_rated_steps={summary['totals']['steps']} "
        f"api_cost_usd={summary['totals']['api_cost_usd']:.6f}",
        flush=True,
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM rating for sampled policy responses.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--step_dir")
    parser.add_argument("--parallel_workers", type=int, default=4)
    parser.add_argument("--step_progress_interval", type=int, default=10)
    parser.add_argument("--refresh_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()

    if args.refresh_only:
        summary = refresh_run_artifacts(run_dir)
        print(
            f"[DONE] run={run_dir.name} refresh_only=1 total_rated_steps={summary['totals']['steps']} "
            f"api_cost_usd={summary['totals']['api_cost_usd']:.6f}",
            flush=True,
        )
        return

    if args.step_dir:
        result = rate_step_payload(
            str(run_dir),
            str(Path(args.step_dir).resolve()),
            str(_default_completed_log()),
            args.step_progress_interval,
        )
        print(
            f"[STEP-DONE] run={result['run_name']} step={result['step_name']} "
            f"valid={result['valid_examples']} skipped={result['skipped_examples']} "
            f"cost_usd={result['api_cost_usd']:.6f}",
            flush=True,
        )
        return

    sys.exit(
        run_rate_for_run(
            run_dir=run_dir,
            parallel_workers=args.parallel_workers,
            step_progress_interval=args.step_progress_interval,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()

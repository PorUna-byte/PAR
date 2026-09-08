from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path).expanduser()
    path = Path(value).expanduser()
    return path.name, path


def step_dirs_for_run(run_dir: Path) -> list[Path]:
    sample_root = run_dir / "sample_on_test"
    if not sample_root.exists():
        return []
    step_dirs = [path for path in sample_root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    step_dirs.sort(key=lambda path: int(path.name.split("_")[-1]) if path.name.split("_")[-1].isdigit() else 0)
    return step_dirs


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def preference_score(
    row: dict[str, Any],
    policy_reward_field: str,
    reference_reward_field: str,
    max_refs: int | None,
) -> float | None:
    policy_reward = to_float(row.get(policy_reward_field))
    if policy_reward is None and policy_reward_field != "policy_reward":
        policy_reward = to_float(row.get("policy_reward"))
    if policy_reward is None:
        return None

    reference_reward = to_float(row.get(reference_reward_field))
    if reference_reward is not None:
        return sigmoid(policy_reward - reference_reward)

    refs = row.get("ref_rewards") or []
    if not isinstance(refs, list) or not refs:
        return None
    if max_refs is not None:
        refs = refs[:max_refs]
    ref_values = [to_float(item) for item in refs]
    ref_values = [item for item in ref_values if item is not None]
    if not ref_values:
        return None
    return sum(sigmoid(policy_reward - ref) for ref in ref_values) / len(ref_values)


def load_run_points(
    run_dir: Path,
    policy_reward_field: str,
    reference_reward_field: str,
    max_refs: int | None,
    latest_only: bool,
) -> list[tuple[float, float]]:
    step_dirs = step_dirs_for_run(run_dir)
    if latest_only and step_dirs:
        step_dirs = [step_dirs[-1]]

    points: list[tuple[float, float]] = []
    for step_dir in step_dirs:
        rated_path = step_dir / "merged_rated.json"
        if not rated_path.exists():
            continue
        rows = load_json(rated_path)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            winrate = to_float(row.get("winrate"))
            if winrate is None:
                continue
            score = preference_score(row, policy_reward_field, reference_reward_field, max_refs)
            if score is None:
                continue
            points.append((score, winrate))
    return points


def binned_points(points: list[tuple[float, float]], bins: int) -> list[dict[str, float | int]]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for score, winrate in points:
        idx = min(max(int(score * bins), 0), bins - 1)
        grouped[idx].append((score, winrate))

    rows: list[dict[str, float | int]] = []
    for idx in sorted(grouped):
        values = grouped[idx]
        rows.append(
            {
                "bin": idx,
                "count": len(values),
                "preference_score": sum(item[0] for item in values) / len(values),
                "winrate": sum(item[1] for item in values) / len(values),
            }
        )
    return rows


def plot(summary: dict[str, Any], output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=180)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#667085", linewidth=1.8, linestyle="--", label="Perfect Calibration")

    for run in summary["runs"]:
        rows = run["bins"]
        if not rows:
            continue
        x = [row["preference_score"] for row in rows]
        y = [row["winrate"] for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.0, label=run["label"])

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Preference Score")
    ax.set_ylabel("Winrate")
    ax.set_title("Winrate vs. Preference Score")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot calibration between reward-model preference score and judged winrate.")
    parser.add_argument("--run", action="append", default=[], help="Run directory, optionally LABEL=/path/to/run_dir. May be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--policy_reward_field", default="policy_reward_origin")
    parser.add_argument("--reference_reward_field", default="reference_reward_origin")
    parser.add_argument("--max_refs", type=int, default=None)
    parser.add_argument("--latest_only", action="store_true")
    args = parser.parse_args()

    if not args.run:
        raise ValueError("At least one --run is required.")

    runs = []
    total_points = 0
    for item in args.run:
        label, run_dir = parse_run_arg(item)
        points = load_run_points(run_dir, args.policy_reward_field, args.reference_reward_field, args.max_refs, args.latest_only)
        total_points += len(points)
        runs.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "points": len(points),
                "bins": binned_points(points, max(args.bins, 1)),
            }
        )

    if total_points == 0:
        raise RuntimeError("No rated calibration points found. Run src/sbatch/05_rating.sh before plotting calibration.")

    summary = {
        "policy_reward_field": args.policy_reward_field,
        "reference_reward_field": args.reference_reward_field,
        "max_refs": args.max_refs,
        "latest_only": args.latest_only,
        "runs": runs,
    }

    output = Path(args.output)
    plot(summary, output)
    if args.summary_output:
        dump_json(Path(args.summary_output), summary)
    print(json.dumps({"output": str(output), "runs": len(runs), "points": total_points}, indent=2))


if __name__ == "__main__":
    main()

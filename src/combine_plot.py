from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ONLINE_ALGORITHMS = {"ppo", "grpo", "a2c"}
METHOD_SUFFIXES = ("_meanstd", "_par", "_vanilla", "_warm")
PLOT_METHODS = ("meanstd", "par", "vanilla", "warm", "dpo")
ANALYSIS_GROUP_PREFIXES = (
    "analysis_principle1",
    "analysis_principle2",
    "analysis_suite",
    "analysis_robust",
    "analysis_dataeffi",
)
CURVE_COLORS = (
    "#155eef",
    "#12b76a",
    "#f79009",
    "#d92d20",
    "#7a5af8",
    "#0086c9",
    "#a15c07",
    "#667085",
    "#c11574",
    "#039855",
    "#6941c6",
    "#b42318",
)
METHOD_STYLE = {
    "meanstd": {"label": "MeanStd", "color": "#155eef", "linestyle": "-", "reward_marker": "o", "win_marker": "s"},
    "par": {"label": "PAR", "color": "#12b76a", "linestyle": "-", "reward_marker": "o", "win_marker": "s"},
    "vanilla": {"label": "Vanilla", "color": "#f79009", "linestyle": "-", "reward_marker": "o", "win_marker": "s"},
    "warm": {"label": "Warm", "color": "#d92d20", "linestyle": "-", "reward_marker": "o", "win_marker": "s"},
    "dpo": {"label": "DPO", "color": "#7a5af8", "linestyle": "-.", "reward_marker": "D", "win_marker": "X"},
}


def src_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return src_root().parent


def default_exp_runs_dir() -> Path:
    return project_root() / "exp_runs"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def identify_method(run_name: str) -> tuple[str | None, str | None]:
    for suffix in METHOD_SUFFIXES:
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)], suffix[1:]
    return None, None


def split_prefix(prefix: str) -> tuple[str | None, str | None]:
    """Split ppo_gemma2-2b_ultrafb_bin into (ppo, gemma2-2b_ultrafb_bin)."""
    if "_" not in prefix:
        return None, None
    algorithm, experiment_key = prefix.split("_", 1)
    if not algorithm or not experiment_key:
        return None, None
    return algorithm, experiment_key


def with_matching_dpo(prefix: str, method_runs: dict[str, Path], groups: dict[str, dict[str, Path]]) -> dict[str, Path]:
    algorithm, experiment_key = split_prefix(prefix)
    if algorithm not in ONLINE_ALGORITHMS or experiment_key is None:
        return method_runs

    dpo_vanilla = groups.get(f"dpo_{experiment_key}", {}).get("vanilla")
    if dpo_vanilla is None:
        return method_runs

    combined = dict(method_runs)
    combined["dpo"] = dpo_vanilla
    return combined


def discover_groups(exp_runs_dir: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    for run_dir in sorted(exp_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        prefix, method = identify_method(run_dir.name)
        if prefix is None or method is None:
            continue
        groups.setdefault(prefix, {})[method] = run_dir
    return groups


def analysis_group_for_run(run_name: str) -> str | None:
    for group_name in ANALYSIS_GROUP_PREFIXES:
        if run_name == group_name or run_name.startswith(f"{group_name}_"):
            return group_name
    return None


def discover_analysis_groups(exp_runs_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {group_name: [] for group_name in ANALYSIS_GROUP_PREFIXES}
    for run_dir in sorted(exp_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        group_name = analysis_group_for_run(run_dir.name)
        if group_name is None:
            continue
        groups[group_name].append(run_dir)
    return {group_name: run_dirs for group_name, run_dirs in groups.items() if run_dirs}


def summary_path(run_dir: Path) -> Path:
    return run_dir / "llm_rating_summary.json"


def extract_curve(summary: dict[str, Any]) -> tuple[list[int], list[float | None], list[float | None]]:
    steps = summary.get("steps", [])
    x = [int(step.get("step_number", 0)) for step in steps]
    rewards = [step.get("avg_proxy_reward_origin") for step in steps]
    wins = [step.get("avg_win_rate") for step in steps]
    return x, rewards, wins


def plot_group(prefix: str, method_runs: dict[str, Path], output_dir: Path) -> Path | None:
    curves: list[tuple[str, list[int], list[float | None], list[float | None]]] = []
    for method in PLOT_METHODS:
        run_dir = method_runs.get(method)
        if run_dir is None:
            continue
        path = summary_path(run_dir)
        if not path.exists():
            print(f"[SKIP] {run_dir.name} missing llm_rating_summary.json")
            continue
        summary = load_json(path)
        x, rewards, wins = extract_curve(summary)
        if not x:
            print(f"[SKIP] {run_dir.name} has no summarized steps")
            continue
        curves.append((method, x, rewards, wins))

    if len(curves) < 2:
        print(f"[SKIP] {prefix} has fewer than 2 comparable method curves")
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_reward, ax_win) = plt.subplots(2, 1, figsize=(9.2, 8.2), dpi=200, sharex=True)

    for method, x, rewards, wins in curves:
        style = METHOD_STYLE[method]
        ax_reward.plot(
            x,
            rewards,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            marker=style["reward_marker"],
            markersize=4.0,
        )
        ax_win.plot(
            x,
            wins,
            label=style["label"],
            color=style["color"],
            linestyle="--",
            linewidth=2.2,
            marker=style["win_marker"],
            markersize=3.8,
        )

    ax_reward.set_title(prefix)
    ax_reward.set_ylabel("Proxy Reward Origin")
    ax_win.set_ylabel("Win Rate")
    ax_win.set_xlabel("Timestep")
    ax_win.set_ylim(0.0, 1.0)

    handles, labels = ax_reward.get_legend_handles_labels()
    ax_reward.legend(handles, labels, loc="best", ncol=2, frameon=True)

    fig.tight_layout()
    output_path = output_dir / f"{prefix}_combined_curve.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def analysis_label(group_name: str, run_name: str) -> str:
    suffix = run_name[len(group_name) :].lstrip("_")
    return (suffix or run_name).replace("_", " ")


def plot_analysis_group(group_name: str, run_dirs: list[Path], output_dir: Path) -> Path | None:
    curves: list[tuple[str, list[int], list[float | None], list[float | None]]] = []
    for run_dir in run_dirs:
        path = summary_path(run_dir)
        if not path.exists():
            print(f"[SKIP] {run_dir.name} missing llm_rating_summary.json")
            continue
        summary = load_json(path)
        x, rewards, wins = extract_curve(summary)
        if not x:
            print(f"[SKIP] {run_dir.name} has no summarized steps")
            continue
        curves.append((analysis_label(group_name, run_dir.name), x, rewards, wins))

    if len(curves) < 2:
        print(f"[SKIP] {group_name} has fewer than 2 analysis curves")
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_reward, ax_win) = plt.subplots(2, 1, figsize=(9.8, 8.6), dpi=200, sharex=True)

    for idx, (label, x, rewards, wins) in enumerate(curves):
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        ax_reward.plot(
            x,
            rewards,
            label=label,
            color=color,
            linestyle="-",
            linewidth=2.2,
            marker="o",
            markersize=3.8,
        )
        ax_win.plot(
            x,
            wins,
            label=label,
            color=color,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            markersize=3.5,
        )

    ax_reward.set_title(group_name.replace("_", " "))
    ax_reward.set_ylabel("Proxy Reward Origin")
    ax_win.set_ylabel("Win Rate")
    ax_win.set_xlabel("Timestep")
    ax_win.set_ylim(0.0, 1.0)

    handles, labels = ax_reward.get_legend_handles_labels()
    ax_reward.legend(handles, labels, loc="best", ncol=2, frameon=True)

    fig.tight_layout()
    output_path = output_dir / f"{group_name}_combined_curve.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine llm rating curves across RL variants and analysis suites.")
    parser.add_argument("--exp_runs_dir", type=Path, default=default_exp_runs_dir())
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Existing method prefix or analysis group name, e.g. ppo_gemma2-2b_ultrafb_bin or analysis_principle1.",
    )
    args = parser.parse_args()

    exp_runs_dir = args.exp_runs_dir.resolve()
    if not exp_runs_dir.exists():
        raise FileNotFoundError(f"exp_runs directory not found: {exp_runs_dir}")

    groups = discover_groups(exp_runs_dir)
    analysis_groups = discover_analysis_groups(exp_runs_dir)
    requested_prefixes = set(args.prefix)
    if requested_prefixes:
        prefixes = sorted(prefix for prefix in requested_prefixes if prefix not in ANALYSIS_GROUP_PREFIXES)
        analysis_prefixes = sorted(prefix for prefix in requested_prefixes if prefix in ANALYSIS_GROUP_PREFIXES)
    else:
        prefixes = sorted(prefix for prefix in groups.keys() if split_prefix(prefix)[0] != "dpo")
        analysis_prefixes = sorted(analysis_groups.keys())

    if not prefixes and not analysis_prefixes:
        print(f"[SKIP] No matching grouped runs found under {exp_runs_dir}")
        return

    generated = 0
    for prefix in prefixes:
        method_runs = groups.get(prefix)
        if not method_runs:
            print(f"[SKIP] No grouped runs found for prefix={prefix}")
            continue
        method_runs = with_matching_dpo(prefix, method_runs, groups)
        output_path = plot_group(prefix, method_runs, exp_runs_dir)
        if output_path is None:
            continue
        generated += 1
        print(f"[DONE] wrote {output_path}")

    for group_name in analysis_prefixes:
        run_dirs = analysis_groups.get(group_name)
        if not run_dirs:
            print(f"[SKIP] No analysis runs found for group={group_name}")
            continue
        output_path = plot_analysis_group(group_name, run_dirs, exp_runs_dir)
        if output_path is None:
            continue
        generated += 1
        print(f"[DONE] wrote {output_path}")

    if generated == 0:
        print("[SKIP] No combined plots were generated")


if __name__ == "__main__":
    main()

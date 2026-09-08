"""Plot average winrate histogram for analysis_principle2 runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SIGMOID_FUNCS = ["tanh", "fittedpoly", "sigmoid", "sigmoidk2", "sigmoidk3"]
FUNC_LABELS = ["tanh", "fittedpoly", "sigmoid", "sigmoidk2", "sigmoidk3"]

COLOR_CENTERED = "#87CEEB"    # sky blue
COLOR_UNCENTERED = "#F4A0A0"  # salmon/light red


def load_avg_winrate(run_dir: Path) -> float | None:
    summary_path = run_dir / "llm_rating_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open() as f:
        data = json.load(f)
    wins = [s["avg_win_rate"] for s in data.get("steps", []) if s.get("avg_win_rate") is not None]
    wins = wins[-7:]
    return sum(wins) / len(wins) if wins else None


def main() -> None:
    exp_runs_dir = Path(__file__).resolve().parents[2] / "exp_runs"

    centered_vals: list[float] = []
    uncentered_vals: list[float] = []

    for func in SIGMOID_FUNCS:
        c = load_avg_winrate(exp_runs_dir / f"analysis_principle2_{func}_centered")
        u = load_avg_winrate(exp_runs_dir / f"analysis_principle2_{func}_uncentered")
        centered_vals.append(c if c is not None else float("nan"))
        uncentered_vals.append(u if u is not None else float("nan"))

    x = np.arange(len(SIGMOID_FUNCS))
    bar_width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    bars_c = ax.bar(x - bar_width / 2, centered_vals, width=bar_width, label="Centered",
                    color=COLOR_CENTERED, zorder=3)
    bars_u = ax.bar(x + bar_width / 2, uncentered_vals, width=bar_width, label="Uncentered",
                    color=COLOR_UNCENTERED, zorder=3)

    all_vals = [v for v in centered_vals + uncentered_vals if not np.isnan(v)]
    y_min = max(0.0, min(all_vals) - 0.05)
    y_max = max(all_vals) + 0.05
    ax.set_ylim(round(y_min - 0.005, 2), round(y_max + 0.005, 2))

    for bar in list(bars_c) + list(bars_u):
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(FUNC_LABELS, fontsize=11)
    ax.set_xlabel("Sigmoid-like function", fontsize=12)
    ax.set_ylabel("Average Winrate", fontsize=12)
    ax.set_title("Average Winrate", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = exp_runs_dir / "analysis_principle2_winrate_histogram.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] saved to {output_path}")


if __name__ == "__main__":
    main()

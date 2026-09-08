import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

CURVE_PACKAGE_DIR = Path(__file__).resolve().parents[1]
JSON_DIR = Path(os.environ.get("CURVE_JSON_DIR", CURVE_PACKAGE_DIR / "json"))

JSON_FILES = [
    JSON_DIR / "ppo_gemma2-2b_ultrafb_bin.json",
    JSON_DIR / "data_efficiency.json",
    JSON_DIR / "bounded_rewards.json",
    JSON_DIR / "robust_warm_minmax_par.json",
]

OUT_DIR = Path(os.environ.get("CURVE_FIGURE_DIR", CURVE_PACKAGE_DIR / "figures" / "final"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINRATE_TOP_K = 2
FADED_ALPHA = 0.55
HIGHLIGHT_ALPHA = 0.95
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]

def format_k(x, pos):
    if x == 0:
        return "0"
    return f"{int(x/1000)}k"


def pretty_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def rank_curves(curves, metric, method="final"):
    ranked = []
    for curve in curves:
        points = sorted(curve["points"], key=lambda p: p["step"])
        vals = [p[metric] for p in points]
        if method == "average":
            score = sum(vals) / len(vals)
        elif method == "final":
            score = vals[-1]
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        ranked.append((curve["curve"], score, points))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def style_curves(ordered_names):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = list(plt.get_cmap("tab10").colors)
    return {
        name: {
            "color": color_cycle[i % len(color_cycle)],
            "marker": MARKERS[i % len(MARKERS)],
        }
        for i, name in enumerate(ordered_names)
    }


def setup_axis(ax, curves):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax.tick_params(axis="both", labelsize=11)

    all_x = [p["step"] for c in curves for p in c["points"]]
    ax.set_xlim(min(all_x), max(all_x))


def plot_metric(
    ax,
    curves,
    ordered_names,
    curve_styles,
    metric,
    linestyle="-",
    highlighted=None,
    fade_unhighlighted=False,
):
    highlighted = highlighted or set()
    curve_lookup = {c["curve"]: c for c in curves}

    for name in reversed(ordered_names):
        curve = curve_lookup[name]
        points = sorted(curve["points"], key=lambda p: p["step"])
        x = [p["step"] for p in points]
        y = [p[metric] for p in points]
        style = curve_styles[name]
        is_highlighted = name in highlighted

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=linestyle,
            linewidth=2.35 if is_highlighted else 1.8,
            markersize=5.0 if is_highlighted else 4.4,
            alpha=(
                HIGHLIGHT_ALPHA
                if is_highlighted or not fade_unhighlighted
                else FADED_ALPHA
            ),
            zorder=3 if is_highlighted else 2,
        )


def save_combined_plot(data, dataset_name, top_k=2):
    curves = data["curves"]
    ranked = rank_curves(curves, "winrate", method="final")
    ordered_names = [name for name, _, _ in ranked]
    highlighted = set(ordered_names[:top_k])
    curve_styles = style_curves(ordered_names)

    fig = plt.figure(figsize=(11.4, 4.15))
    gridspec = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 0.255, 1.0),
        left=0.075,
        right=0.985,
        bottom=0.16,
        top=0.88,
        wspace=0.015,
    )
    axes = [fig.add_subplot(gridspec[0, 0])]
    legend_ax = fig.add_subplot(gridspec[0, 1])
    axes.append(fig.add_subplot(gridspec[0, 2], sharex=axes[0]))
    legend_ax.axis("off")

    plot_metric(
        axes[0],
        curves,
        ordered_names,
        curve_styles,
        "proxy_reward",
        linestyle="-",
    )
    plot_metric(
        axes[1],
        curves,
        ordered_names,
        curve_styles,
        "winrate",
        linestyle="--",
        highlighted=highlighted,
        fade_unhighlighted=True,
    )

    axes[0].set_title("(a) Proxy Reward", fontsize=12, pad=8)
    axes[1].set_title("(b) Winrate", fontsize=12, pad=8)
    axes[0].set_ylabel("Proxy Reward", fontsize=12)
    axes[1].set_ylabel("Winrate", fontsize=12)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_label_coords(1.08, 0.5)
    axes[1].yaxis.tick_right()

    for ax in axes:
        ax.set_xlabel("Steps", fontsize=12)
        setup_axis(ax, curves)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=curve_styles[name]["color"],
            marker=curve_styles[name]["marker"],
            linestyle="-",
            linewidth=2.1,
            markersize=5.0,
        )
        for name in ordered_names
    ]
    legend = legend_ax.legend(
        legend_handles,
        ordered_names,
        loc="center",
        frameon=True,
        fontsize=10.5,
        title_fontsize=10.5,
        labelspacing=0.58,
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.5,
    )
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(0.8)

    base = f"{Path(dataset_name.replace(' ', '_')).stem}"
    png_path = OUT_DIR / f"{base}.png"
    pdf_path = OUT_DIR / f"{base}.pdf"

    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    generated = []
    for json_file in JSON_FILES:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_name = pretty_name(json_file)
        generated.extend(save_combined_plot(data, dataset_name, WINRATE_TOP_K))

    print("Generated files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

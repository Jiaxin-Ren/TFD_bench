"""
Reliability Diagram / 可靠性图

Visualize model calibration by comparing predicted confidence with actual accuracy.
通过对比预测置信度和实际准确率来可视化模型校准。

Usage / 使用方法:
    from analysis.visualization import reliability
    fig = reliability.plot_reliability_diagram(confidences, accuracies)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.visualization.io import (
    discover_prediction_runs,
    display_method,
    primary_probs,
    save_figure,
)

from analysis.visualization.style import (
    ANNOTATION_SIZE,
    DOUBLE_COLUMN_MM,
    LINE_WIDTH,
    SUBMISSION_DPI,
    figure_size,
)

def compute_calibration_bins(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration statistics for bins.
    计算分箱的校准统计量。
    
    Args:
        confidences: Predicted confidence scores [0, 1] / 预测置信度
        correctness: Binary correctness (1=correct, 0=wrong) / 是否正确
        n_bins: Number of bins / 分箱数
    
    Returns:
        bin_centers: Center of each bin / 每个分箱的中心
        bin_accuracies: Accuracy in each bin / 每个分箱的准确率
        bin_confidences: Mean confidence in each bin / 每个分箱的平均置信度
        bin_counts: Number of samples in each bin / 每个分箱的样本数
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if i == n_bins - 1:  # Include right boundary for last bin
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(correctness[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
    
    return bin_centers, bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    figsize: Tuple[int, int] = (6, 5),
    color: str = "#356AC3",
    show_gap: bool = True,
    show_counts: bool = False,
    show_legend: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot reliability diagram (calibration plot).
    绘制可靠性图（校准图）。
    
    Args:
        confidences: Predicted confidence scores [0, 1]
        correctness: Binary correctness (1=correct, 0=wrong)
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size
        color: Bar color
        show_gap: Show calibration gap
        show_counts: Show sample counts
        show_legend: Show the legend on this axis
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    standalone = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    bin_centers, bin_accuracies, bin_confidences, bin_counts = compute_calibration_bins(
        confidences, correctness, n_bins
    )

    bin_width = 1.0 / n_bins
    valid = bin_counts > 0
    ax.bar(
        bin_centers[valid],
        bin_accuracies[valid],
        width=bin_width * 0.88,
        color=color,
        edgecolor="#1F3D68",
        linewidth=0.8,
        label="Accuracy",
        zorder=2,
    )

    if show_gap:
        gap_bottom = np.minimum(bin_accuracies[valid], bin_confidences[valid])
        gap_height = np.abs(bin_accuracies[valid] - bin_confidences[valid])
        ax.bar(
            bin_centers[valid],
            gap_height,
            bottom=gap_bottom,
            width=bin_width * 0.88,
            facecolor="white",
            edgecolor="#FF646A",
            linewidth=1.1,
            hatch="///",
            label="Calibration gap",
            zorder=3,
        )

    ax.plot(
        [0, 1], [0, 1], "k--", linewidth=LINE_WIDTH,
        label="Perfect calibration", zorder=4,
    )

    if show_counts:
        for center, count in zip(bin_centers, bin_counts):
            if count > 0:
                ax.text(
                    center, 0.02, f"{int(count)}",
                    ha="center", va="bottom", fontsize=ANNOTATION_SIZE, color="gray",
                )

    total_samples = np.sum(bin_counts)
    ece = float(
        np.sum(bin_counts[valid] * np.abs(bin_accuracies[valid] - bin_confidences[valid]))
        / total_samples
    ) if total_samples else float("nan")

    ax.text(
        0.97, 0.04, f"ECE={ece * 100:.2f}%",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=ANNOTATION_SIZE,
        bbox={"facecolor": "white", "edgecolor": "#555555", "pad": 2.0},
        zorder=5,
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, zorder=0)

    if standalone:
        fig.tight_layout()
    return fig


def plot_multi_reliability(
    results_dict: dict,
    n_bins: int = 10,
    figsize: Tuple[int, int] = None,
    ncols: int = 5,
    title: str = "Reliability Diagrams Comparison"
) -> plt.Figure:
    """
    Plot multiple reliability diagrams for comparison.
    绘制多个可靠性图进行对比。
    
    Args:
        results_dict: Dict of {method_name: (confidences, correctness)}
        n_bins: Number of bins
        figsize: Figure size (auto if None)
        ncols: Number of columns
        title: Overall title
    
    Returns:
        matplotlib Figure
    """
    if not results_dict:
        raise ValueError("No calibration predictions were provided.")
    n_methods = len(results_dict)
    ncols = min(ncols, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    if figsize is None:
        figsize = figure_size(DOUBLE_COLUMN_MM, 125)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (method_name, (confidences, correctness)) in enumerate(results_dict.items()):
        plot_reliability_diagram(
            confidences, correctness,
            n_bins=n_bins,
            title=method_name,
            color="#356AC3",
            show_counts=False,
            show_legend=False,
            ax=axes[i],
        )
        if i // ncols != nrows - 1:
            axes[i].set_xlabel("")
        if i % ncols != 0:
            axes[i].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    legend_order = ("Accuracy", "Calibration gap", "Perfect calibration")
    ordered_handles = [legend_items[label] for label in legend_order]

    unused_axes = axes[n_methods:]
    for unused_ax in unused_axes:
        unused_ax.set_axis_off()
    if len(unused_axes):
        unused_axes[-1].legend(
            ordered_handles, legend_order, loc="lower right", frameon=False,
        )
    else:
        fig.legend(
            ordered_handles, legend_order, loc="lower right", frameon=False,
        )

    fig.tight_layout()
    return fig


def generate_reliability_plot(
    results_dir: str | Path,
    dataset: str,
    backbone: str,
    *,
    methods: list[str] | None = None,
    config: str = "clean",
    bins: int = 10,
) -> plt.Figure:
    runs = discover_prediction_runs(
        results_dir, dataset=dataset, backbone=backbone, methods=methods, config=config
    )
    plot_data = {}
    for method, method_runs in runs.items():
        confidences, correctness = [], []
        for _, arrays in method_runs:
            probs = primary_probs(arrays)
            targets = arrays["id_targets"].astype(int)
            confidences.append(probs.max(axis=-1))
            correctness.append((probs.argmax(axis=-1) == targets).astype(float))
        plot_data[display_method(method)] = (
            np.concatenate(confidences),
            np.concatenate(correctness),
        )
    return plot_multi_reliability(
        plot_data,
        n_bins=bins,
        ncols=min(5, len(plot_data)),
        title=f"{dataset.upper()} / {backbone} / {config} — Reliability",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real reliability diagrams.")
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    parser.add_argument("--dataset", default="seu")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "results" / "figures" / "reliability.png"),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()
    fig = generate_reliability_plot(
        args.results_dir,
        args.dataset,
        args.backbone,
        methods=args.methods,
        config=args.config,
        bins=args.bins,
    )
    path = save_figure(fig, args.output, args.dpi)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

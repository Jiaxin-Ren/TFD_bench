"""
Comparison Charts / 对比图表

Multi-method comparison bar charts and heatmaps.
多方法对比柱状图和热力图。

Usage / 使用方法:
    from analysis.visualization import comparison
    fig = comparison.plot_metric_comparison(results_dict)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.visualization.io import display_method, load_summary_group, save_figure

from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    LINE_WIDTH,
    SINGLE_COLUMN_MM,
    SUBMISSION_DPI,
    figure_size,
    method_color,
)

SUMMARY_METRICS = {
    "test/cls/Acc": ("ACC", True),
    "test/cal/ECE": ("ECE", False),
    "ood/AUROC": ("AUROC", True),
}


def plot_metric_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison",
    figsize: Tuple[int, int] = None,
    highlight_best: bool = True,
    higher_better: Dict[str, bool] = None,
) -> plt.Figure:
    """
    Plot grouped bar chart comparing methods across metrics.
    绘制分组柱状图对比各方法的指标。
    
    Args:
        results_dict: Dict of {method_name: {metric_name: value}}
        metrics: List of metric names to plot (auto-detect if None)
        title: Plot title
        figsize: Figure size
        highlight_best: Highlight best value for each metric
        higher_better: Dict of {metric_name: True/False}
    
    Returns:
        matplotlib Figure
    """
    # Get methods and metrics
    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    if figsize is None:
        figsize = figure_size(DOUBLE_COLUMN_MM, 80)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Default higher_better
    if higher_better is None:
        higher_better = {m: not any(x in m.lower() for x in ['loss', 'error', 'fpr', 'nll', 'ece']) 
                        for m in metrics}
    
    # Prepare data
    x = np.arange(n_methods)
    width = 0.8 / n_metrics
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
    
    # Find best values
    best_values = {}
    for metric in metrics:
        values = [results_dict[m].get(metric, 0) for m in methods]
        if higher_better.get(metric, True):
            best_values[metric] = max(values)
        else:
            best_values[metric] = min(values)
    
    # Plot bars
    for i, metric in enumerate(metrics):
        values = [results_dict[m].get(metric, 0) for m in methods]
        offset = (i - n_metrics/2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Highlight best
        if highlight_best:
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val == best_values[metric]:
                    bar.set_edgecolor('gold')
                    bar.set_linewidth(2)
    
    # Labels
    ax.set_xlabel('Method')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_metric_heatmap(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison Heatmap",
    figsize: Tuple[int, int] = None,
    cmap: str = "RdYlGn",
    annotate: bool = True,
    higher_better: Dict[str, bool] = None,
) -> plt.Figure:
    """
    Plot heatmap comparing methods across metrics.
    绘制热力图对比各方法的指标。
    
    Args:
        results_dict: Dict of {method_name: {metric_name: value}}
        metrics: List of metric names
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        annotate: Show values in cells
        higher_better: Dict of {metric_name: True/False}
    
    Returns:
        matplotlib Figure
    """
    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    if figsize is None:
        figsize = figure_size(DOUBLE_COLUMN_MM, 100)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Default higher_better
    if higher_better is None:
        higher_better = {m: not any(x in m.lower() for x in ['loss', 'error', 'fpr', 'nll', 'ece']) 
                        for m in metrics}
    
    # Build data matrix
    data = np.zeros((n_methods, n_metrics))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            data[i, j] = results_dict[method].get(metric, 0)
    
    # Normalize each column (metric) to [0, 1] for visualization
    data_normalized = np.zeros_like(data)
    for j, metric in enumerate(metrics):
        col = data[:, j]
        if col.max() > col.min():
            normalized = (col - col.min()) / (col.max() - col.min())
            if not higher_better.get(metric, True):
                normalized = 1 - normalized  # Flip so "good" is always high
            data_normalized[:, j] = normalized
        else:
            data_normalized[:, j] = 0.5
    
    # Plot heatmap
    im = ax.imshow(data_normalized, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Annotate
    if annotate:
        for i in range(n_methods):
            for j in range(n_metrics):
                val = data[i, j]
                text_color = 'white' if data_normalized[i, j] < 0.3 or data_normalized[i, j] > 0.7 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                       color=text_color)
    
    # Labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_methods))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(methods)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Performance (higher = better)')
    
    plt.tight_layout()
    return fig


def plot_radar_chart(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison (Radar)",
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Plot radar chart comparing methods.
    绘制雷达图对比各方法。
    
    Args:
        results_dict: Dict of {method_name: {metric_name: value}}
        metrics: List of metric names
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
    
    n_metrics = len(metrics)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    if figsize is None:
        figsize = figure_size(SINGLE_COLUMN_MM, SINGLE_COLUMN_MM)
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Normalize data
    data_matrix = np.array([[results_dict[m].get(metric, 0) for metric in metrics] for m in methods])
    data_normalized = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8)
    
    for i, method in enumerate(methods):
        values = data_normalized[i].tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=LINE_WIDTH, color=method_color(method), label=method)
        ax.fill(angles, values, alpha=0.1, color=method_color(method))
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    return fig


def generate_comparison_plot(
    summary: str | Path,
    dataset: str,
    backbone: str,
    config: str = "clean",
    methods: list[str] | None = None,
) -> plt.Figure:
    """Create three independent panels from an aggregated summary file."""
    records = load_summary_group(
        summary, dataset=dataset, backbone=backbone, config=config, methods=methods
    )
    if not records:
        raise FileNotFoundError(
            f"No summary records for {dataset}/{backbone}/{config} in {summary}"
        )

    method_names = list(records)
    y = np.arange(len(method_names))
    fig, axes = plt.subplots(
        1, 3, figsize=figure_size(DOUBLE_COLUMN_MM, 100), sharey=True
    )
    for ax, (metric, (label, higher_better)) in zip(axes, SUMMARY_METRICS.items()):
        means = np.asarray([
            float(records[name][metric]["mean"]) * 100
            for name in method_names
        ])
        stds = np.asarray([
            float(records[name][metric].get("std", 0.0) or 0.0) * 100
            for name in method_names
        ])
        ax.errorbar(means, y, xerr=stds, fmt="o", capsize=2.5, color="#2878b5")
        best = int(np.argmax(means) if higher_better else np.argmin(means))
        ax.scatter(means[best], y[best], s=85, color="#f2c14e", edgecolor="#7a5c00", zorder=3)
        ax.set_title(f"{label} {'↑' if higher_better else '↓'}")
        ax.set_xlabel("Percent (%)")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([display_method(name) for name in method_names])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ACC, ECE, and AUROC from summary.json.")
    parser.add_argument("--summary", default=str(_PROJECT_ROOT / "results" / "summary.json"))
    parser.add_argument("--dataset", default="seu")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "results" / "figures" / "comparison.png"),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()
    fig = generate_comparison_plot(
        args.summary, args.dataset, args.backbone, args.config, args.methods
    )
    path = save_figure(fig, args.output, args.dpi)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

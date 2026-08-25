"""
Uncertainty Distribution Plots / 不确定性分布图

Compare uncertainty distributions between ID and OOD samples.
对比 ID 和 OOD 样本的不确定性分布。

Usage / 使用方法:
    from analysis.visualization import uncertainty
    fig = uncertainty.plot_uncertainty_distribution(id_scores, ood_scores)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.visualization.io import (
    discover_prediction_runs,
    display_method,
    finite_ood_score_pair,
    save_figure,
)


def plot_uncertainty_distribution(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    title: str = "Uncertainty Distribution",
    id_label: str = "In-Distribution",
    ood_label: str = "Out-of-Distribution",
    xlabel: str = "Uncertainty Score",
    figsize: Tuple[int, int] = (8, 5),
    bins: int = 50,
    alpha: float = 0.6,
    id_color: str = "#3498db",
    ood_color: str = "#e74c3c",
    show_stats: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot uncertainty score distributions for ID vs OOD.
    绘制 ID 和 OOD 的不确定性分数分布。
    
    Args:
        id_scores: Uncertainty scores for in-distribution samples
        ood_scores: Uncertainty scores for out-of-distribution samples
        title: Plot title
        id_label: Label for ID distribution
        ood_label: Label for OOD distribution
        xlabel: X-axis label
        figsize: Figure size
        bins: Number of histogram bins
        alpha: Transparency
        id_color: Color for ID distribution
        ood_color: Color for OOD distribution
        show_stats: Show mean and std in legend
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    # Compute statistics
    id_mean, id_std = np.mean(id_scores), np.std(id_scores)
    ood_mean, ood_std = np.mean(ood_scores), np.std(ood_scores)
    
    # Create labels with stats
    if show_stats:
        id_label_full = f"{id_label}\n(μ={id_mean:.3f}, σ={id_std:.3f})"
        ood_label_full = f"{ood_label}\n(μ={ood_mean:.3f}, σ={ood_std:.3f})"
    else:
        id_label_full = id_label
        ood_label_full = ood_label
    
    # Plot histograms
    ax.hist(id_scores, bins=bins, alpha=alpha, color=id_color, 
            label=id_label_full, density=True, edgecolor='white')
    ax.hist(ood_scores, bins=bins, alpha=alpha, color=ood_color,
            label=ood_label_full, density=True, edgecolor='white')
    
    # Add vertical lines for means
    ax.axvline(id_mean, color=id_color, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(ood_mean, color=ood_color, linestyle='--', linewidth=2, alpha=0.8)
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_multi_uncertainty_comparison(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = None,
    ncols: int = 2,
    title: str = "Uncertainty Distributions by Method"
) -> plt.Figure:
    """
    Plot uncertainty distributions for multiple methods.
    绘制多个方法的不确定性分布对比。
    
    Args:
        results_dict: Dict of {method_name: (id_scores, ood_scores)}
        figsize: Figure size
        ncols: Number of columns
        title: Overall title
    
    Returns:
        matplotlib Figure
    """
    n_methods = len(results_dict)
    ncols = min(ncols, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    
    for i, (method_name, (id_scores, ood_scores)) in enumerate(results_dict.items()):
        if i < len(axes):
            plot_uncertainty_distribution(
                id_scores, ood_scores,
                title=method_name,
                ax=axes[i]
            )
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_violin_comparison(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Uncertainty Score Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot violin plots comparing ID and OOD scores across methods.
    绘制小提琴图对比各方法的 ID 和 OOD 分数。
    
    Args:
        results_dict: Dict of {method_name: (id_scores, ood_scores)}
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    methods = list(results_dict.keys())
    positions = []
    data = []
    colors = []
    labels = []
    
    id_color = "#3498db"
    ood_color = "#e74c3c"
    
    for i, method in enumerate(methods):
        id_scores, ood_scores = results_dict[method]
        
        # ID scores
        data.append(id_scores)
        positions.append(i * 3)
        colors.append(id_color)
        labels.append(f"{method}\n(ID)")
        
        # OOD scores
        data.append(ood_scores)
        positions.append(i * 3 + 1)
        colors.append(ood_color)
        labels.append(f"{method}\n(OOD)")
    
    # Create violin plot
    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    
    # Set x-axis ticks
    tick_positions = [i * 3 + 0.5 for i in range(len(methods))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(methods, fontsize=10)
    
    # Labels
    ax.set_ylabel("Uncertainty Score", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=id_color, alpha=0.6, label='In-Distribution'),
        Patch(facecolor=ood_color, alpha=0.6, label='Out-of-Distribution')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def generate_uncertainty_plot(
    results_dir: str | Path,
    dataset: str,
    backbone: str,
    *,
    methods: list[str] | None = None,
    config: str = "clean",
) -> plt.Figure:
    runs = discover_prediction_runs(
        results_dir, dataset=dataset, backbone=backbone, methods=methods, config=config
    )
    plot_data = {}
    for method, method_runs in runs.items():
        pairs = [finite_ood_score_pair(arrays) for _, arrays in method_runs]
        id_scores = [pair[0] for pair in pairs if pair[0].size]
        ood_scores = [pair[1] for pair in pairs if pair[1].size]
        if id_scores and ood_scores:
            plot_data[display_method(method)] = (
                np.concatenate(id_scores), np.concatenate(ood_scores)
            )
    if not plot_data:
        raise ValueError("No finite ID/OOD scores are available for plotting.")
    return plot_multi_uncertainty_comparison(
        plot_data,
        ncols=min(3, len(plot_data)),
        title=f"{dataset.upper()} / {backbone} / {config} — Native OOD Scores",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real ID/OOD score distributions.")
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    parser.add_argument("--dataset", default="seu")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "results" / "figures" / "ood_scores.png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    fig = generate_uncertainty_plot(
        args.results_dir,
        args.dataset,
        args.backbone,
        methods=args.methods,
        config=args.config,
    )
    path = save_figure(fig, args.output, args.dpi)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

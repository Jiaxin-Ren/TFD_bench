"""
ROC Curve Comparison / ROC曲线对比

Compare OOD detection performance across methods using ROC curves.
使用ROC曲线对比各方法的OOD检测性能。

Usage / 使用方法:
    from analysis.visualization import roc
    fig = roc.plot_roc_curves(results_dict)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.visualization.io import (
    discover_prediction_runs,
    display_method,
    finite_ood_score_pair,
    save_figure,
)


def plot_roc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves for OOD Detection",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.
    绘制多个方法的ROC曲线。
    
    Args:
        results_dict: Dict of {method_name: (labels, scores)}
            labels: Binary labels (0=ID, 1=OOD)
            scores: OOD scores (higher = more OOD)
        title: Plot title
        figsize: Figure size
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (method_name, (labels, scores)) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i % 10], linewidth=2,
                label=f'{method_name} (AUC = {roc_auc:.4f})')
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    # Labels
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_pr_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall Curves for OOD Detection",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple methods.
    绘制多个方法的精确率-召回率曲线。
    
    Args:
        results_dict: Dict of {method_name: (labels, scores)}
        title: Plot title
        figsize: Figure size
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (method_name, (labels, scores)) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        
        ax.plot(recall, precision, color=colors[i % 10], linewidth=2,
                label=f'{method_name} (AP = {ap:.4f})')
    
    # Labels
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_roc_and_pr(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "OOD Detection Performance",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot both ROC and PR curves side by side.
    并排绘制ROC和PR曲线。
    
    Args:
        results_dict: Dict of {method_name: (labels, scores)}
        title: Overall title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plot_roc_curves(results_dict, title="ROC Curves", ax=ax1)
    plot_pr_curves(results_dict, title="Precision-Recall Curves", ax=ax2)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def compute_fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    Compute FPR at a given TPR threshold.
    计算给定TPR阈值下的FPR。
    
    Args:
        labels: Binary labels
        scores: OOD scores
        target_tpr: Target TPR (default 0.95)
    
    Returns:
        FPR at target TPR
    """
    labels = np.asarray(labels).reshape(-1).astype(bool)
    scores = np.asarray(scores).reshape(-1)
    positives = labels.sum()
    negatives = (~labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_idx = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(labels) - 1]
    true_positive = np.cumsum(sorted_labels)[threshold_idx]
    false_positive = threshold_idx + 1 - true_positive
    recall = true_positive / positives
    cutoff = np.flatnonzero(recall >= target_tpr)[0]
    return float(false_positive[cutoff] / negatives)


def plot_roc_pr_runs(runs: dict, title: str) -> plt.Figure:
    """Plot seed-wise curves with the mean and one-standard-deviation band."""
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(14, 5.5))
    roc_grid = np.linspace(0, 1, 301)
    recall_grid = np.linspace(0, 1, 301)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(runs), 1)))

    for color, (method, method_runs) in zip(colors, sorted(runs.items())):
        tprs, precisions, aucs, aps, fpr95s = [], [], [], [], []
        skipped_seeds = []
        for seed, arrays in method_runs:
            id_scores, ood_scores = finite_ood_score_pair(arrays)
            if id_scores.size == 0 or ood_scores.size == 0:
                skipped_seeds.append(seed)
                continue
            labels = np.r_[np.zeros(id_scores.size), np.ones(ood_scores.size)]
            scores = np.r_[id_scores, ood_scores]
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            roc_ax.plot(fpr, tpr, color=color, linewidth=0.7, alpha=0.22)
            pr_ax.plot(recall, precision, color=color, linewidth=0.7, alpha=0.22)
            tprs.append(np.interp(roc_grid, fpr, tpr))
            precisions.append(np.interp(recall_grid, recall[::-1], precision[::-1]))
            aucs.append(auc(fpr, tpr))
            aps.append(average_precision_score(labels, scores))
            fpr95s.append(compute_fpr_at_tpr(labels, scores))

        if skipped_seeds:
            print(f"Skipped {method} ROC seeds with no finite ID/OOD scores: {skipped_seeds}")
        if not tprs:
            print(f"Skipped {method} ROC: no valid runs")
            continue

        tprs = np.asarray(tprs)
        precisions = np.asarray(precisions)
        roc_mean, roc_std = tprs.mean(axis=0), tprs.std(axis=0)
        pr_mean, pr_std = precisions.mean(axis=0), precisions.std(axis=0)
        label = (
            f"{display_method(method)} "
            f"(AUROC={np.mean(aucs):.3f}, FPR95={np.mean(fpr95s):.3f})"
        )
        roc_ax.plot(roc_grid, roc_mean, color=color, linewidth=2, label=label)
        roc_ax.fill_between(
            roc_grid,
            np.clip(roc_mean - roc_std, 0, 1),
            np.clip(roc_mean + roc_std, 0, 1),
            color=color,
            alpha=0.12,
        )
        pr_ax.plot(
            recall_grid,
            pr_mean,
            color=color,
            linewidth=2,
            label=f"{display_method(method)} (AUPR={np.mean(aps):.3f})",
        )
        pr_ax.fill_between(
            recall_grid,
            np.clip(pr_mean - pr_std, 0, 1),
            np.clip(pr_mean + pr_std, 0, 1),
            color=color,
            alpha=0.12,
        )

    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    for ax, x_label, y_label, panel in (
        (roc_ax, "False Positive Rate", "True Positive Rate", "ROC"),
        (pr_ax, "Recall", "Precision", "Precision–Recall"),
    ):
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=x_label, ylabel=y_label, title=panel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def generate_roc_plot(
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
    return plot_roc_pr_runs(
        runs, f"{dataset.upper()} / {backbone} / {config} — OOD Detection"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real OOD ROC and PR curves.")
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    parser.add_argument("--dataset", default="seu")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "results" / "figures" / "roc_pr.png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    fig = generate_roc_plot(
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

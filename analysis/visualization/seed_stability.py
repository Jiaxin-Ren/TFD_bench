"""Show per-seed ACC, ECE, and OOD AUROC from saved predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.visualization.io import (
    discover_prediction_runs,
    display_method,
    finite_ood_score_pair,
    primary_probs,
    save_figure,
)
from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    SUBMISSION_DPI,
    figure_size,
)



def _ece(probs: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def _run_metrics(arrays: dict) -> tuple[float, float, float]:
    probs = primary_probs(arrays, "id")
    targets = arrays["id_targets"].astype(int)
    acc = float((probs.argmax(axis=1) == targets).mean())
    ece = _ece(probs, targets)
    id_scores, ood_scores = finite_ood_score_pair(arrays)
    if id_scores.size == 0 or ood_scores.size == 0:
        auroc = float("nan")
        return acc, ece, auroc
    labels = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    auroc = float(roc_auc_score(labels, np.r_[id_scores, ood_scores]))
    return acc, ece, auroc


def generate_seed_stability_plot(
    results_dir: str | Path,
    dataset: str,
    backbone: str,
    config: str = "clean",
    methods: list[str] | None = None,
) -> plt.Figure:
    runs = discover_prediction_runs(
        results_dir, dataset=dataset, backbone=backbone, methods=methods, config=config
    )
    method_names = list(runs)
    y = np.arange(len(method_names))
    fig, axes = plt.subplots(
        1, 3, figsize=figure_size(DOUBLE_COLUMN_MM, 100), sharey=True
    )
    for method_index, method in enumerate(method_names):
        values = np.asarray([_run_metrics(arrays) for _, arrays in runs[method]]) * 100
        for metric_index, ax in enumerate(axes):
            jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(values[:, metric_index], method_index + jitter, alpha=0.7, s=28)
            ax.scatter(values[:, metric_index].mean(), method_index, marker="D", color="black", s=32)
    for ax, label in zip(axes, ("ACC ↑", "ECE ↓", "AUROC ↑")):
        ax.set(title=label, xlabel="Percent (%)", yticks=y)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    axes[0].set_yticklabels([display_method(name) for name in method_names])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    parser.add_argument("--dataset", default="seu")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "results" / "figures" / "seed_stability.png"),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()
    fig = generate_seed_stability_plot(args.results_dir, args.dataset, args.backbone, args.config, args.methods)
    save_figure(fig, args.output, args.dpi)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

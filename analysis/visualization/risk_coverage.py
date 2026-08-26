"""Plot selective-classification risk versus coverage from saved predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.visualization.io import (
    discover_prediction_runs,
    display_method,
    primary_probs,
    save_figure,
)

from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    LINE_WIDTH,
    SEED_LINE_WIDTH,
    SUBMISSION_DPI,
    figure_size,
    method_color,
)

def _risk_coverage(
    probs: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the curve by accepting the least uncertain predictions first."""
    uncertainty = np.asarray(uncertainty).reshape(-1)
    if len(uncertainty) != len(targets):
        raise ValueError("Expected one native uncertainty score per target.")
    finite = np.isfinite(uncertainty)
    probs = probs[finite]
    targets = targets[finite]
    uncertainty = uncertainty[finite]
    if not len(targets):
        raise ValueError("No finite native uncertainty scores are available.")
    errors = (probs.argmax(axis=1) != targets).astype(float)
    order = np.argsort(uncertainty, kind="stable")
    errors = errors[order]
    coverage = np.arange(1, len(errors) + 1, dtype=float) / len(errors)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return coverage, risk, float(np.trapz(risk, coverage))


def generate_risk_coverage_plot(
    results_dir: str | Path,
    dataset: str,
    backbone: str,
    config: str = "clean",
    methods: list[str] | None = None,
) -> plt.Figure:
    runs = discover_prediction_runs(
        results_dir,
        dataset=dataset,
        backbone=backbone,
        methods=methods,
        config=config,
    )
    grid = np.linspace(0.01, 1.0, 200)
    fig, ax = plt.subplots(figsize=figure_size(DOUBLE_COLUMN_MM, 105))
    for method, seed_runs in runs.items():
        curves = []
        for _, arrays in seed_runs:
            coverage, risk, _ = _risk_coverage(
                primary_probs(arrays, "id"),
                arrays["id_targets"].astype(int),
                arrays["id_ood_scores"],
            )
            curve = np.interp(grid, coverage, risk, left=risk[0], right=risk[-1])
            curves.append(curve)
            ax.plot(coverage, risk, color=method_color(method), alpha=0.15, linewidth=SEED_LINE_WIDTH)
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        ddof = 1 if len(values) > 1 else 0
        std = values.std(axis=0, ddof=ddof)
        ax.plot(grid, mean, color=method_color(method), linewidth=LINE_WIDTH, label=display_method(method))
        ax.fill_between(grid, np.maximum(0, mean - std), mean + std, color=method_color(method), alpha=0.12)
    ax.set(xlabel="Coverage", ylabel="Risk")
    ax.grid(alpha=0.3)
    ax.legend(title="Method", ncol=2)
    fig.tight_layout()
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
        default=str(_PROJECT_ROOT / "results" / "figures" / "risk_coverage.png"),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()
    fig = generate_risk_coverage_plot(
        args.results_dir, args.dataset, args.backbone, args.config, args.methods
    )
    path = save_figure(
        fig,
        args.output,
        args.dpi,
    )
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

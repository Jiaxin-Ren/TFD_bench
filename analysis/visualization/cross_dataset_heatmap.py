"""Cross-dataset relative-quality heatmaps for clean TFD-Bench results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.methods import PLOT_METHODS, display_method, plot_method_sort_key
from analysis.visualization.cross_dataset_profile import DATASET_ORDER
from analysis.visualization.io import save_figure
from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    SUBMISSION_DPI,
    figure_size,
)


HEATMAP_METRICS = (
    ("test/cls/Acc", "ACC", True),
    ("test/cls/NLL", "NLL", False),
    ("test/cal/ECE", "ECE", False),
    ("ood/AUROC", "AUROC", True),
    ("test/sc/AURC", "AURC", False),
)

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "cyan_white_magenta",
    ("#59D3D0", "#FBFBFB", "#F06ADE"),
)
HEATMAP_CMAP.set_bad("#E6E6E6")


def _mean_metric(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def load_metric_matrices(
    summary: dict,
    *,
    backbone: str,
    datasets: tuple[str, ...],
    methods: list[str] | None = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return complete method-by-dataset matrices for each metric."""

    allowed_methods = set(methods) if methods else set(PLOT_METHODS)
    records: dict[tuple[str, str], dict] = {}
    for record in summary.values():
        dataset = record.get("dataset")
        method = record.get("method")
        if (
            dataset in datasets
            and method in allowed_methods
            and record.get("backbone") == backbone
            and record.get("config", "clean") == "clean"
        ):
            records[(method, dataset)] = record.get("metrics", {})

    common_methods = []
    for method in sorted(allowed_methods, key=plot_method_sort_key):
        complete = True
        for dataset in datasets:
            metrics = records.get((method, dataset), {})
            if any(_mean_metric(metrics, key) is None for key, _, _ in HEATMAP_METRICS):
                complete = False
                break
        if complete:
            common_methods.append(method)

    if not common_methods:
        raise ValueError("No method has complete clean metrics across all datasets.")

    matrices: dict[str, np.ndarray] = {}
    for key, label, _ in HEATMAP_METRICS:
        matrices[label] = np.asarray(
            [
                [
                    _mean_metric(records[(method, dataset)], key)
                    for dataset in datasets
                ]
                for method in common_methods
            ],
            dtype=float,
        )
    return common_methods, matrices


def relative_quality(values: np.ndarray, *, higher_better: bool) -> np.ndarray:
    """Normalize one metric globally over all method-dataset entries."""

    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, np.nan, dtype=float)

    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if np.isclose(lower, upper):
        quality = np.full_like(values, 0.5, dtype=float)
    else:
        quality = (values - lower) / (upper - lower)
    if not higher_better:
        quality = 1.0 - quality
    quality[~finite] = np.nan
    return quality


def generate_cross_dataset_heatmap(
    summary: dict,
    *,
    backbone: str = "resnet",
    datasets: tuple[str, ...] = DATASET_ORDER,
    methods: list[str] | None = None,
) -> plt.Figure:
    """Create five metric heatmaps plus one shared colorbar column."""

    datasets = tuple(datasets)
    if len(datasets) != 6:
        raise ValueError("The cross-dataset heatmap requires exactly six datasets.")

    method_names, matrices = load_metric_matrices(
        summary,
        backbone=backbone,
        datasets=datasets,
        methods=methods,
    )

    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_MM, 82))
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=(1.28, 1.0, 1.0, 1.0, 1.0, 0.07),
        wspace=0.13,
    )

    image = None
    for index, (_, label, higher_better) in enumerate(HEATMAP_METRICS):
        ax = fig.add_subplot(grid[0, index])
        quality = relative_quality(
            matrices[label],
            higher_better=higher_better,
        )
        image = ax.imshow(
            quality,
            cmap=HEATMAP_CMAP,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        direction = "↑" if higher_better else "↓"
        ax.set_title(f"{label} {direction}", pad=5)
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(
            [dataset.upper() for dataset in datasets],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(len(method_names)))
        if index == 0:
            ax.set_yticklabels([display_method(method) for method in method_names])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    colorbar_ax = fig.add_subplot(grid[0, 5])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks((0.0, 0.5, 1.0))
    colorbar.set_label("Relative quality (higher = better)", labelpad=5)

    fig.subplots_adjust(
        left=0.075,
        right=0.94,
        bottom=0.16,
        top=0.93,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-dataset clean relative-quality heatmaps."
    )
    parser.add_argument(
        "--summary",
        default=str(_PROJECT_ROOT / "results" / "summary.json"),
    )
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument(
        "--datasets",
        nargs=6,
        default=list(DATASET_ORDER),
    )
    parser.add_argument("--methods", nargs="*")
    parser.add_argument(
        "--output",
        default=str(
            _PROJECT_ROOT
            / "results"
            / "figures"
            / "cross_dataset_heatmap.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()

    with Path(args.summary).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    figure = generate_cross_dataset_heatmap(
        summary,
        backbone=args.backbone,
        datasets=tuple(args.datasets),
        methods=args.methods,
    )
    path = save_figure(figure, args.output, args.dpi)
    plt.close(figure)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

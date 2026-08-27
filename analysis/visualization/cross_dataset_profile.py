"""Cross-dataset method profiles for the clean TFD-Bench setting.

The large radar chart reports the macro average across datasets. The six
smaller radar charts retain the dataset-level profiles. Each metric is
normalized globally across methods and datasets, and every axis is oriented
so that larger values indicate better relative quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.methods import PLOT_METHODS, display_method, plot_method_sort_key
from analysis.visualization.io import save_figure
from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    LEGEND_SIZE,
    LINE_WIDTH,
    SUBMISSION_DPI,
    figure_size,
)


DATASET_ORDER = ("seu", "wt", "pu", "xjtu", "hit", "cwru")

FAMILY_METHODS = (
    ("post-hoc calibration", ("temperature_scaling",)),
    ("Bayesian inference", ("variational_bnn", "sgld", "sghmc")),
    (
        "ensemble",
        (
            "deep_ensemble",
            "snapshot_ensemble",
            "batch_ensemble",
            "packed_ensemble",
        ),
    ),
    (
        "posterior approximation",
        ("swag", "laplace_approx", "mc_dropout", "mc_batch_norm"),
    ),
    ("evidential learning", ("edl",)),
)

# Color-blind-safe colors are paired with unique line and marker styles so
# that the profiles remain distinguishable in grayscale journal printing.
PROFILE_STYLES = {
    "baseline": {
        "color": "#202020",
        "linestyle": (0, (4.0, 2.0)),
        "marker": "o",
    },
    "post-hoc calibration": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "s",
    },
    "Bayesian inference": {
        "color": "#E69F00",
        "linestyle": (0, (5.0, 1.5)),
        "marker": "^",
    },
    "ensemble": {
        "color": "#009E73",
        "linestyle": (0, (3.0, 1.2, 1.0, 1.2)),
        "marker": "D",
    },
    "posterior approximation": {
        "color": "#CC79A7",
        "linestyle": (0, (1.2, 1.2)),
        "marker": "v",
    },
    "evidential learning": {
        "color": "#D55E00",
        "linestyle": (0, (6.0, 1.2, 1.0, 1.2)),
        "marker": "P",
    },
}


PROFILE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("test/cls/Acc", "ACC", True),
    ("test/cls/NLL", "NLL", False),
    ("test/cls/Brier", "Brier", False),
    ("test/cal/ECE", "ECE", False),
    ("ood/AUROC", "AUROC", True),
    ("test/sc/AURC", "AURC", False),
)


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


def _profile_style(method: str) -> dict:
    if method == "max_softmax":
        return PROFILE_STYLES["baseline"]
    for family, family_methods in FAMILY_METHODS:
        if method in family_methods:
            return PROFILE_STYLES[family]
    return PROFILE_STYLES["baseline"]


def load_profile_data(
    summary: dict,
    *,
    backbone: str,
    datasets: Sequence[str],
    methods: Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load complete clean profiles grouped by dataset and raw method name."""

    allowed_methods = (
        set(methods) | {"max_softmax"} if methods else set(PLOT_METHODS)
    )
    profiles: dict[str, dict[str, np.ndarray]] = {
        dataset: {} for dataset in datasets
    }
    for record in summary.values():
        dataset = record.get("dataset")
        method = record.get("method")
        if (
            dataset not in profiles
            or record.get("backbone") != backbone
            or record.get("config", "clean") != "clean"
            or method not in allowed_methods
        ):
            continue

        metrics = record.get("metrics", {})
        values = []
        for key, _, _ in PROFILE_METRICS:
            value = _mean_metric(metrics, key)
            if value is None:
                break
            values.append(value)
        if len(values) == len(PROFILE_METRICS):
            profiles[dataset][method] = np.asarray(values, dtype=float)

    missing_datasets = [dataset for dataset, values in profiles.items() if not values]
    if missing_datasets:
        raise ValueError(
            "No complete clean profiles for: " + ", ".join(missing_datasets)
        )

    common_methods = set.intersection(
        *(set(profiles[dataset]) for dataset in datasets)
    )
    if not common_methods:
        raise ValueError("No method has a complete profile across all datasets.")

    ordered_profiles = {
        dataset: {
            method: profiles[dataset][method]
            for method in sorted(common_methods, key=plot_method_sort_key)
        }
        for dataset in datasets
    }

    # Put heterogeneous metrics on a common relative-quality scale. The
    # normalization is global for each metric, so values remain comparable
    # across datasets; all axes point outwards for better performance.
    for metric_index, (_, _, higher_better) in enumerate(PROFILE_METRICS):
        metric_values = np.asarray(
            [
                values[metric_index]
                for dataset_profiles in ordered_profiles.values()
                for values in dataset_profiles.values()
            ],
            dtype=float,
        )
        lower = float(np.min(metric_values))
        upper = float(np.max(metric_values))
        span = upper - lower
        for dataset_profiles in ordered_profiles.values():
            for values in dataset_profiles.values():
                quality = 0.5 if np.isclose(span, 0.0) else (
                    values[metric_index] - lower
                ) / span
                values[metric_index] = quality if higher_better else 1.0 - quality

    return ordered_profiles


def _angles() -> np.ndarray:
    count = len(PROFILE_METRICS)
    return np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)


def _configure_radar(ax: plt.Axes, *, label_size: float) -> None:
    angles = _angles()
    closed_angles = np.concatenate((angles, angles[:1]))
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [label for _, label, _ in PROFILE_METRICS],
        fontsize=label_size,
    )
    ax.tick_params(axis="x", pad=3)
    ax.set_ylim(0.0, 1.04)
    ax.set_yticks(())
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    grid_color = "#d2d5d8"
    for radius in np.linspace(0.2, 1.0, 5):
        ax.plot(
            closed_angles,
            np.full_like(closed_angles, radius),
            color=grid_color,
            linewidth=0.55,
            zorder=0,
        )
        ax.text(
            0.0,
            radius,
            f"{int(radius * 100)}",
            ha="left",
            va="bottom",
            fontsize=max(label_size - 1.5, 4.8),
            color="#62666a",
            zorder=1,
        )
    for angle in angles:
        ax.plot(
            (angle, angle),
            (0.0, 1.0),
            color=grid_color,
            linewidth=0.55,
            zorder=0,
        )


def _plot_profiles(
    ax: plt.Axes,
    profiles: dict[str, np.ndarray],
    *,
    linewidth: float,
    alpha: float,
    markersize: float,
    best_method: str,
) -> None:
    angles = _angles()
    closed_angles = np.concatenate((angles, angles[:1]))
    for method, values in profiles.items():
        is_best = method == best_method
        style = _profile_style(method)
        closed_values = np.concatenate((values, values[:1]))
        ax.plot(
            closed_angles,
            closed_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=linewidth * 1.55 if is_best else linewidth,
            alpha=1.0 if is_best else alpha,
            marker=style["marker"],
            markersize=markersize * 1.15 if is_best else markersize,
            markerfacecolor=style["color"] if is_best else "white",
            markeredgecolor=style["color"],
            markeredgewidth=0.65,
            zorder=3 if is_best else 2,
        )


def generate_cross_dataset_profile(
    summary: dict,
    *,
    backbone: str = "resnet",
    datasets: Sequence[str] = DATASET_ORDER,
    methods: Sequence[str] | None = None,
) -> plt.Figure:
    """Create radar profiles for MSP and one representative per UQ family."""

    datasets = tuple(datasets)
    if len(datasets) != 6:
        raise ValueError("The cross-dataset profile requires exactly six datasets.")

    profiles = load_profile_data(
        summary,
        backbone=backbone,
        datasets=datasets,
        methods=methods,
    )
    all_method_names = list(next(iter(profiles.values())))
    all_macro_profiles = {
        method: np.mean(
            np.stack([profiles[dataset][method] for dataset in datasets]),
            axis=0,
        )
        for method in all_method_names
    }
    method_score = {
        method: float(np.mean(all_macro_profiles[method]))
        for method in all_method_names
    }

    if methods:
        method_names = sorted(all_method_names, key=plot_method_sort_key)
    else:
        baseline = "max_softmax"
        representatives = []
        for _, family_methods in FAMILY_METHODS:
            available = [
                method for method in family_methods if method in all_method_names
            ]
            if available:
                representatives.append(
                    max(
                        available,
                        key=lambda method: (
                            method_score[method],
                            -plot_method_sort_key(method)[0],
                        ),
                    )
                )
        method_names = (
            [baseline] if baseline in all_method_names else []
        ) + representatives

    best_method = max(method_names, key=method_score.__getitem__)

    profiles = {
        dataset: {
            method: profiles[dataset][method]
            for method in method_names
        }
        for dataset in datasets
    }
    macro_profiles = {
        method: all_macro_profiles[method]
        for method in method_names
    }

    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_MM, 92))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.65, 1.0, 1.0, 1.0),
        wspace=0.34,
        hspace=0.52,
    )

    macro_ax = fig.add_subplot(grid[:, 0], projection="polar")
    _configure_radar(macro_ax, label_size=8)
    _plot_profiles(
        macro_ax,
        macro_profiles,
        linewidth=LINE_WIDTH,
        alpha=0.94,
        markersize=2.7,
        best_method=best_method,
    )
    macro_ax.set_title("Cross-dataset mean", pad=11)

    for index, dataset in enumerate(datasets):
        row, column = divmod(index, 3)
        ax = fig.add_subplot(grid[row, column + 1], projection="polar")
        _configure_radar(ax, label_size=6)
        _plot_profiles(
            ax,
            profiles[dataset],
            linewidth=max(LINE_WIDTH - 0.15, 0.65),
            alpha=0.88,
            markersize=1.8,
            best_method=best_method,
        )
        ax.set_title(dataset.upper(), pad=7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=_profile_style(method)["color"],
            linestyle=_profile_style(method)["linestyle"],
            linewidth=LINE_WIDTH * 1.55 if method == best_method else LINE_WIDTH,
            marker=_profile_style(method)["marker"],
            markersize=2.7,
            markerfacecolor=(
                _profile_style(method)["color"]
                if method == best_method
                else "white"
            ),
            markeredgecolor=_profile_style(method)["color"],
            markeredgewidth=0.55,
            label=display_method(method),
        )
        for method in method_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=7,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.95,
        bottom=0.16,
        top=0.85,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot macro and dataset-level clean method profiles."
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
            / "cross_dataset_profile.png"
        ),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()

    with Path(args.summary).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    figure = generate_cross_dataset_profile(
        summary,
        backbone=args.backbone,
        datasets=args.datasets,
        methods=args.methods,
    )
    path = save_figure(figure, args.output, args.dpi)
    plt.close(figure)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

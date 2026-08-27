"""Generate benchmark comparison figures from ``results/summary.json``.

Aggregate figures use ``summary.json``. If current per-seed prediction artifacts
are available, this entry point also produces diagnostic figures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.generate_tables import METHOD_NAMES
from analysis.methods import PLOT_METHODS, method_display_sort_key
from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    LINE_WIDTH,
    SUBMISSION_DPI,
    figure_size,
    method_color,
)
from analysis.visualization.cross_dataset_heatmap import (
    generate_cross_dataset_heatmap,
)
from analysis.visualization.cross_dataset_profile import (
    generate_cross_dataset_profile,
)
from analysis.visualization.reliability import generate_reliability_plot
from analysis.visualization.risk_coverage import generate_risk_coverage_plot
from analysis.visualization.roc import generate_roc_plot
from analysis.visualization.seed_stability import generate_seed_stability_plot
from analysis.visualization.uncertainty import generate_uncertainty_plot


PLOT_METRICS = {
    "test/cls/Acc": {"label": "ACC", "scale": 100, "higher_better": True, "unit": "Percent (%)"},
    "test/cal/ECE": {"label": "ECE", "scale": 100, "higher_better": False, "unit": "Percent (%)"},
    "ood/AUROC": {"label": "AUROC", "scale": 100, "higher_better": True, "unit": "Percent (%)"},
}

SELECTIVE_METRIC = {
    "test/sc/AURC": {
        "label": "AURC",
        "scale": 100,
        "higher_better": False,
        "unit": "Percent (%)",
    },
}



def load_results(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _config_sort_key(config: str) -> tuple[int, str, int]:
    if config == "clean":
        return (0, "", 0)
    if config == "operating_shift":
        return (1, "", 0)
    match = re.fullmatch(r"(.+)_s(\d+)", config)
    if match:
        return (2, match.group(1), int(match.group(2)))
    return (3, config, 0)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _group_results(results: dict, metric_definitions: dict = PLOT_METRICS) -> dict:
    """Return method metrics grouped by dataset/backbone/test configuration."""
    groups: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for stats in results.values():
        metrics = stats.get("metrics", {})
        values = {}
        for metric, config in metric_definitions.items():
            metric_stats = metrics.get(metric)
            if metric_stats is None:
                break
            mean = metric_stats.get("mean") if isinstance(metric_stats, dict) else metric_stats
            if mean is None:
                break
            std = metric_stats.get("std", 0.0) if isinstance(metric_stats, dict) else 0.0
            values[config["label"]] = {
                "mean": float(mean) * config["scale"],
                "std": float(std or 0.0) * config["scale"],
            }
        if len(values) != len(metric_definitions):
            continue

        group_key = (
            stats.get("dataset", "unknown"),
            stats.get("backbone", "unknown"),
            stats.get("config", "clean"),
        )
        raw_method = stats.get("method", "unknown")
        method = METHOD_NAMES.get(raw_method, raw_method)
        groups[group_key][method] = values
    return {
        group_key: dict(
            sorted(methods.items(), key=lambda item: method_display_sort_key(item[0]))
        )
        for group_key, methods in groups.items()
    }


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_metric_panels(
    methods: dict[str, dict[str, float]],
    title: str,
    metric_definitions: dict = PLOT_METRICS,
) -> plt.Figure:
    """Plot each metric on an independent axis so every scale is legible."""
    method_names = sorted(methods, key=method_display_sort_key)
    y = np.arange(len(method_names))
    fig, axes = plt.subplots(
        1,
        len(metric_definitions),
        figsize=figure_size(DOUBLE_COLUMN_MM, 100),
        sharey=True,
    )

    for ax, metric_config in zip(axes, metric_definitions.values()):
        label = metric_config["label"]
        values = np.asarray([methods[method][label]["mean"] for method in method_names])
        errors = np.asarray([methods[method][label]["std"] for method in method_names])
        ax.errorbar(
            values,
            y,
            xerr=errors,
            fmt="o",
            markersize=5,
            capsize=2.5,
            color="#3498db",
            ecolor="#8ebfe0",
            zorder=3,
        )
        best_index = int(np.argmax(values) if metric_config["higher_better"] else np.argmin(values))
        ax.scatter(
            values[best_index],
            y[best_index],
            s=90,
            facecolor="#f1c40f",
            edgecolor="#8a6d00",
            linewidth=1.2,
            zorder=4,
            label="Best",
        )

        spread = float(values.max() - values.min())
        margin = max(spread * 0.08, 0.05)
        ax.set_xlim(float(values.min() - margin), float(values.max() + margin))
        direction = "↑" if metric_config["higher_better"] else "↓"
        ax.set_title(f"{label} {direction}")
        ax.set_xlabel(metric_config["unit"])
        ax.set_yticks(y)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    axes[0].set_yticklabels(method_names)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_noise_trends(
    groups: dict,
    output_dir: Path,
    dpi: int,
    metric_definitions: dict = PLOT_METRICS,
    suffix: str = "",
) -> int:
    by_benchmark = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    clean = defaultdict(dict)
    for (dataset, backbone, config), methods in groups.items():
        if config == "clean":
            clean[(dataset, backbone)].update(methods)
            continue
        match = re.fullmatch(r"(.+)_s(\d+)", config)
        if not match:
            continue
        for method, values in methods.items():
            by_benchmark[(dataset, backbone)][match.group(1)][method][int(match.group(2))] = values

    count = 0
    for (dataset, backbone), noise_types in sorted(by_benchmark.items()):
        for noise, methods in sorted(noise_types.items()):
            fig, axes = plt.subplots(
                1,
                len(metric_definitions) + 1,
                figsize=figure_size(DOUBLE_COLUMN_MM, 68),
                sharex=True,
            )
            metric_axes = axes[:-1]
            legend_ax = axes[-1]
            for method in sorted(methods, key=method_display_sort_key):
                severities = dict(methods[method])
                color = method_color(next((raw for raw, short in METHOD_NAMES.items() if short == method), method))
                if method in clean[(dataset, backbone)]:
                    severities[0] = clean[(dataset, backbone)][method]
                x = np.asarray(sorted(severities))
                for ax, metric_config in zip(metric_axes, metric_definitions.values()):
                    label = metric_config["label"]
                    means = np.asarray([severities[s][label]["mean"] for s in x])
                    stds = np.asarray([severities[s][label]["std"] for s in x])
                    ax.plot(x, means, marker="o", linewidth=LINE_WIDTH, color=color, label=method)
                    ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.08)
            for metric_index, (ax, metric_config) in enumerate(
                zip(metric_axes, metric_definitions.values())
            ):
                direction = "↑" if metric_config["higher_better"] else "↓"
                ax.set(
                    title=f"{metric_config['label']} {direction}",
                    xlabel="Noise severity",
                    ylabel=metric_config["unit"] if metric_index == 0 else "",
                )
                ax.yaxis.labelpad = 2
                ax.set_xticks(range(6))
                ax.grid(True, alpha=0.3)
            handles, labels = metric_axes[0].get_legend_handles_labels()
            legend_ax.set_axis_off()
            legend_ax.legend(handles, labels, loc="center", frameon=False)
            fig.subplots_adjust(
                left=0.055,
                right=0.995,
                bottom=0.18,
                top=0.92,
                wspace=0.32,
            )
            path = output_dir / _safe_name(dataset) / _safe_name(backbone) / f"noise_{_safe_name(noise)}{suffix}.png"
            _save_figure(fig, path, dpi)
            count += 1
    return count


def generate_all_plots(
    results: dict,
    output_dir: str | Path,
    dpi: int = SUBMISSION_DPI,
    results_dir: str | Path | None = None,
    exclude_methods: set[str] | None = None,
) -> None:
    excluded = set(exclude_methods or ())
    results = {
        key: stats for key, stats in results.items()
        if stats.get("method") in PLOT_METHODS
        and stats.get("method") not in excluded
    }
    output_path = Path(output_dir)
    groups = _group_results(results)
    selective_groups = _group_results(results, SELECTIVE_METRIC)
    if not groups and not selective_groups:
        raise ValueError("No benchmark metrics were found in the summary file.")

    trend_metrics = PLOT_METRICS | SELECTIVE_METRIC
    trend_groups = _group_results(results, trend_metrics)
    figure_count = 0

    cross_dataset_figures = (
        ("cross_dataset_heatmap.png", generate_cross_dataset_heatmap),
        ("cross_dataset_profile.png", generate_cross_dataset_profile),
    )
    for filename, generator in cross_dataset_figures:
        try:
            figure = generator(results)
        except (KeyError, ValueError) as error:
            print(f"Skipped {filename}: {error}")
            continue
        _save_figure(figure, output_path / filename, dpi)
        figure_count += 1

    figure_count += _plot_noise_trends(
        trend_groups,
        output_path,
        dpi,
        metric_definitions=trend_metrics,
    )
    artifact_root = Path(results_dir) if results_dir else _PROJECT_ROOT / "results"
    for (dataset, backbone, config), _ in sorted(
        selective_groups.items(),
        key=lambda item: (item[0][0], item[0][1], _config_sort_key(item[0][2])),
    ):
        group_dir = output_path / _safe_name(dataset) / _safe_name(backbone)
        risk_methods = sorted({
            stats.get("method")
            for stats in results.values()
            if stats.get("dataset") == dataset
            and stats.get("backbone") == backbone
            and stats.get("config", "clean") == config
            and stats.get("method")
        })
        filename = f"{_safe_name(config)}_risk_coverage.png"
        try:
            fig = generate_risk_coverage_plot(
                artifact_root,
                dataset,
                backbone,
                config=config,
                methods=risk_methods,
            )
        except (FileNotFoundError, KeyError, ValueError) as error:
            print(f"Skipped {dataset}/{backbone}/{filename}: {error}")
            continue
        _save_figure(fig, group_dir / filename, dpi)
        figure_count += 1

    distribution_diagnostics = (
        ("reliability", generate_reliability_plot),
        ("ood_scores", generate_uncertainty_plot),
    )
    diagnostic_groups = set(groups) | set(selective_groups)
    for dataset, backbone, config in sorted(
        diagnostic_groups,
        key=lambda key: (key[0], key[1], _config_sort_key(key[2])),
    ):
        group_dir = output_path / _safe_name(dataset) / _safe_name(backbone)
        diagnostic_methods = sorted({
            stats.get("method")
            for stats in results.values()
            if stats.get("dataset") == dataset
            and stats.get("backbone") == backbone
            and stats.get("config", "clean") == config
            and stats.get("method")
        })
        for suffix, generator in distribution_diagnostics:
            filename = f"{_safe_name(config)}_{suffix}.png"
            try:
                fig = generator(
                    artifact_root,
                    dataset,
                    backbone,
                    config=config,
                    methods=diagnostic_methods,
                )
            except (FileNotFoundError, KeyError, ValueError) as error:
                print(f"Skipped {dataset}/{backbone}/{filename}: {error}")
                continue
            _save_figure(fig, group_dir / filename, dpi)
            figure_count += 1

    clean_benchmarks = sorted({(dataset, backbone) for dataset, backbone, config in groups if config == "clean"})
    diagnostics = (
        ("clean_roc_pr.png", generate_roc_plot),
        ("clean_seed_stability.png", generate_seed_stability_plot),
    )
    for dataset, backbone in clean_benchmarks:
        group_dir = output_path / _safe_name(dataset) / _safe_name(backbone)
        diagnostic_methods = sorted({
            stats.get("method")
            for stats in results.values()
            if stats.get("dataset") == dataset
            and stats.get("backbone") == backbone
            and stats.get("config", "clean") == "clean"
            and stats.get("method")
        })
        for filename, generator in diagnostics:
            try:
                fig = generator(
                    artifact_root,
                    dataset,
                    backbone,
                    config="clean",
                    methods=diagnostic_methods,
                )
            except FileNotFoundError as error:
                print(f"Skipped {dataset}/{backbone}/{filename}: {error}")
                continue
            except (KeyError, ValueError) as error:
                print(f"Skipped {dataset}/{backbone}/{filename}: {error}")
                continue
            _save_figure(fig, group_dir / filename, dpi)
            figure_count += 1
    print(f"Generated {figure_count} figures in {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(_PROJECT_ROOT / "results" / "summary.json"))
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    parser.add_argument(
        "--output", default=str(_PROJECT_ROOT / "results" / "figures")
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    parser.add_argument(
        "--exclude-methods",
        nargs="*",
        default=[],
        help="Additional method identifiers to exclude from every figure",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously generated benchmark PNG files before plotting",
    )
    args = parser.parse_args()

    if args.clean:
        output = Path(args.output)
        removed = 0
        if output.exists():
            for path in output.rglob("*.png"):
                path.unlink()
                removed += 1
        print(f"Removed {removed} previous benchmark figures")
    excluded = set(args.exclude_methods)
    if excluded:
        print(f"Excluded methods: {', '.join(sorted(excluded))}")
    generate_all_plots(
        load_results(args.results),
        args.output,
        args.dpi,
        args.results_dir,
        excluded,
    )


if __name__ == "__main__":
    main()

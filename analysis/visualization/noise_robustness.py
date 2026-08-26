"""Plot each noise type separately so unrelated severities are never connected."""

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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.methods import PLOT_METHODS, plot_method_sort_key
from analysis.visualization.io import display_method, save_figure
from analysis.visualization.style import (
    DOUBLE_COLUMN_MM,
    LINE_WIDTH,
    SUBMISSION_DPI,
    figure_size,
    method_color,
)

METRICS = {
    "test/cls/Acc": ("ACC", True),
    "test/cal/ECE": ("ECE", False),
    "ood/AUROC": ("AUROC", True),
}


def _load_noise_data(
    summary_path: str | Path,
    dataset: str,
    backbone: str,
    methods: list[str] | None,
) -> dict:
    with Path(summary_path).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    selected = set(methods or ())
    data = defaultdict(lambda: defaultdict(dict))
    clean = {}
    for record in summary.values():
        method = record.get("method")
        if method not in PLOT_METHODS:
            continue
        if record.get("dataset") != dataset or record.get("backbone") != backbone:
            continue
        if selected and method not in selected:
            continue
        config = record.get("config", "clean")
        if config == "clean":
            clean[method] = record["metrics"]
            continue
        match = re.fullmatch(r"(.+)_s([1-5])", config)
        if match:
            data[match.group(1)][method][int(match.group(2))] = record["metrics"]
    for noise_methods in data.values():
        for method in noise_methods:
            if method in clean:
                noise_methods[method][0] = clean[method]
    return dict(data)


def plot_noise_type(noise: str, methods: dict, dataset: str, backbone: str) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 3, figsize=figure_size(DOUBLE_COLUMN_MM, 75)
    )
    for method, severities in sorted(methods.items(), key=lambda item: plot_method_sort_key(item[0])):
        x = np.asarray(sorted(severities))
        for ax, (metric, (label, _)) in zip(axes, METRICS.items()):
            valid = [severity for severity in x if metric in severities[severity]]
            if not valid:
                continue
            means = np.asarray([severities[s][metric]["mean"] for s in valid]) * 100
            stds = np.asarray([severities[s][metric].get("std", 0.0) or 0.0 for s in valid]) * 100
            ax.plot(valid, means, marker="o", linewidth=LINE_WIDTH, color=method_color(method), label=display_method(method))
            ax.fill_between(valid, means - stds, means + stds, color=method_color(method), alpha=0.1)
    for ax, (_, (label, higher)) in zip(axes, METRICS.items()):
        ax.set(title=f"{label} {'↑' if higher else '↓'}", xlabel="Severity (0 = clean)", ylabel="Percent (%)")
        ax.set_xticks(range(6))
        ax.grid(alpha=0.3)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def generate_noise_plots(
    summary: str | Path,
    output_dir: str | Path,
    dataset: str,
    backbone: str,
    methods: list[str] | None = None,
    noise_type: str | None = None,
    dpi: int = SUBMISSION_DPI,
) -> list[Path]:
    data = _load_noise_data(summary, dataset, backbone, methods)
    if noise_type:
        data = {noise_type: data[noise_type]} if noise_type in data else {}
    if not data:
        raise ValueError(f"No matching noise results for {dataset}/{backbone}.")
    outputs = []
    for noise, noise_methods in sorted(data.items()):
        path = Path(output_dir) / f"noise_{noise}.png"
        save_figure(plot_noise_type(noise, noise_methods, dataset, backbone), path, dpi)
        plt.close("all")
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(_PROJECT_ROOT / "results" / "summary.json"))
    parser.add_argument("--dataset", default="seu")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--noise-type")
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "results" / "figures" / "noise"),
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    args = parser.parse_args()
    for path in generate_noise_plots(
        args.summary, args.output_dir, args.dataset, args.backbone,
        args.methods, args.noise_type, args.dpi
    ):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

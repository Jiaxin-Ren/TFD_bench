"""Shared result loading helpers for standalone visualization commands."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


from analysis.methods import PLOT_METHODS, display_method, plot_method_sort_key
from analysis.visualization.style import SUBMISSION_DPI

def parse_seed(path: Path) -> int:
    match = re.fullmatch(r"seed(\d+)", path.parents[1].name)
    return int(match.group(1)) if match else 0


def discover_prediction_runs(
    results_dir: str | Path,
    *,
    dataset: str,
    backbone: str,
    methods: Iterable[str] | None = None,
    config: str = "clean",
) -> dict[str, list[tuple[int, dict[str, np.ndarray]]]]:
    """Load per-seed prediction artifacts grouped by method."""
    root = Path(results_dir) / dataset / backbone
    selected = set(methods or ())
    grouped: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = defaultdict(list)
    if not root.exists():
        raise FileNotFoundError(f"Result directory does not exist: {root}")

    for path in sorted(root.glob("*/seed*/predictions/*.npz")):
        method = path.parents[2].name
        if method not in PLOT_METHODS:
            continue
        if selected and method not in selected:
            continue
        artifact_config = path.stem
        if method == "temperature_scaling":
            if artifact_config.startswith(("baseline_", "before_")):
                continue
            artifact_config = artifact_config.removeprefix("after_")
        if artifact_config != config:
            continue
        with np.load(path, allow_pickle=False) as archive:
            arrays = {}
            for key in archive.files:
                try:
                    arrays[key] = archive[key]
                except ValueError as error:
                    if "Object arrays cannot be loaded" not in str(error):
                        raise
                    # Legacy artifacts saved optional string metadata as an
                    # object array. Plot inputs are numeric, so ignore only
                    # that unsafe optional field while keeping pickle disabled.
                    continue
        grouped[method].append((parse_seed(path), arrays))

    for runs in grouped.values():
        runs.sort(key=lambda item: item[0])
    if not grouped:
        raise FileNotFoundError(
            f"No prediction artifacts for {dataset}/{backbone}/{config}. "
            "Rerun the experiments with the current code first."
        )
    return dict(sorted(grouped.items(), key=lambda item: plot_method_sort_key(item[0])))


def load_summary_group(
    summary_path: str | Path,
    *,
    dataset: str,
    backbone: str,
    config: str = "clean",
    methods: Iterable[str] | None = None,
) -> dict[str, dict]:
    """Load one dataset/backbone/config group from ``summary.json``."""
    with Path(summary_path).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    selected = set(methods or ())
    group = {}
    for stats in summary.values():
        method = stats.get("method", "unknown")
        if method not in PLOT_METHODS:
            continue
        if (
            stats.get("dataset") == dataset
            and stats.get("backbone") == backbone
            and stats.get("config", "clean") == config
            and (not selected or method in selected)
        ):
            group[method] = stats.get("metrics", {})
    if not group:
        raise ValueError(f"No summary records for {dataset}/{backbone}/{config}.")
    return dict(sorted(group.items(), key=lambda item: plot_method_sort_key(item[0])))


def primary_probs(arrays: dict[str, np.ndarray], split: str = "id") -> np.ndarray:
    probs = np.asarray(arrays[f"{split}_probs"])
    if probs.ndim == 1 or probs.shape[-1] == 1:
        positive = probs.reshape(-1, 1)
        probs = np.concatenate((1 - positive, positive), axis=-1)
    return probs


def native_ood_scores(
    arrays: dict[str, np.ndarray], split: str
) -> np.ndarray:
    """Return native OOD scores, repairing legacy conformal artifacts.

    Older conformal artifacts stored normalized set indicators. Empty sets were
    consequently saved as all-NaN rows. Their cardinality can still be recovered:
    positive finite entries are members and an all-NaN row is an empty set.
    """
    prediction_set_key = f"{split}_prediction_sets"
    if prediction_set_key in arrays:
        prediction_sets = np.asarray(arrays[prediction_set_key])
        members = np.isfinite(prediction_sets) & (prediction_sets > 0)
        return members.sum(axis=-1).astype(float).reshape(-1)
    score_key = "id_ood_scores" if split == "id" else "ood_scores"
    return np.asarray(arrays[score_key]).reshape(-1)


def finite_ood_score_pair(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite ID/OOD score arrays without mixing their class labels."""
    id_scores = native_ood_scores(arrays, "id")
    ood_scores = native_ood_scores(arrays, "ood")
    return id_scores[np.isfinite(id_scores)], ood_scores[np.isfinite(ood_scores)]


def save_figure(fig, output: str | Path, dpi: int = SUBMISSION_DPI) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path

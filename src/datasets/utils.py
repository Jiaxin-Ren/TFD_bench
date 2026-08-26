# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
import copy
from collections.abc import Callable
import numpy as np
import pandas as pd
from typing import Any

import torch
from torch.utils.data import Dataset, random_split


def create_train_val_split(
    dataset: Dataset,
    val_split_rate: float,
    val_transforms: Callable | None = None,
    seed: int = 12345,
) -> tuple[Dataset, Dataset]:
    """Split a dataset for training and validation.

    Args:
        dataset (Dataset): The dataset to be split.
        val_split_rate (float): The amount of the original dataset to use as validation split.
        val_transforms (Callable | None, optional): The transformations to apply on the validation set.
            Defaults to ``None``.
        seed: Fixed split seed, independent from the model-training seed.

    Returns:
        tuple[Dataset, Dataset]: The training and the validation splits.
    """
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(
        dataset,
        [1 - val_split_rate, val_split_rate],
        generator=generator,
    )
    val = copy.deepcopy(val)  # Ensure train.dataset.transform is not modified next line

    val.dataset.transform = val_transforms
    return train, val

def subsample_uniform(
    data: list[Any],
    labels: list[int],
    max_items: int | None,
) -> tuple[list[Any], list[int]]:
    """Select at most max_items elements uniformly and deterministically."""
    if len(data) != len(labels):
        raise ValueError("Data and labels must have identical lengths.")
    if max_items is None or len(data) <= max_items:
        return data, labels
    if max_items <= 0:
        raise ValueError("max_items must be positive or None.")

    positions = np.linspace(
        0, len(data) - 1, max_items
    ).round().astype(int)
    if len(np.unique(positions)) != max_items:
        raise RuntimeError("Uniform subsampling produced duplicate positions.")
    return (
        [data[position] for position in positions],
        [labels[position] for position in positions],
    )


def temporal_window_split(
    data: list[np.ndarray],
    labels: list[int],
    source: str,
    val_ratio: float,
    test_ratio: float = 0.2,
    gap_windows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one recording chronologically with gaps at both boundaries.

    The source recording is split before recordings are concatenated, so every
    source contributes to all three sets without randomly mixing adjacent
    windows. Window indices are retained for auditing and regression tests.
    """
    if len(data) != len(labels):
        raise ValueError(f"Data/label length mismatch for {source}.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one.")
    if gap_windows < 0:
        raise ValueError("gap_windows must be non-negative.")

    size = len(data)
    test_start = int(size * (1 - test_ratio))
    pool_stop = max(test_start - gap_windows, 0)
    val_start = int(round(pool_stop * (1 - val_ratio)))
    train_stop = max(val_start - gap_windows, 0)

    if train_stop == 0 or val_start >= pool_stop or test_start >= size:
        raise ValueError(
            f"Recording {source} has only {size} windows, which is too short "
            "for the requested temporal train/val/test split."
        )

    def frame(indices: range) -> pd.DataFrame:
        positions = list(indices)
        return pd.DataFrame(
            {
                "data": [data[index] for index in positions],
                "label": [labels[index] for index in positions],
                "source": [source] * len(positions),
                "window_index": positions,
            }
        )

    train = frame(range(0, train_stop))
    val = frame(range(val_start, pool_stop))
    test = frame(range(test_start, size))
    return train, val, test


def concatenate_temporal_splits(
    splits: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate per-recording temporal splits without changing their order."""
    if not splits:
        raise ValueError("At least one recording is required for splitting.")
    combined = []
    for position in range(3):
        combined.append(
            pd.concat(
                [split[position] for split in splits],
                ignore_index=True,
            )
        )
    return combined[0], combined[1], combined[2]


def assert_disjoint_temporal_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Raise when an audited source/window identifier occurs in two splits."""
    identifier_sets = []
    for frame in (train, val, test):
        required = {"source", "window_index"}
        if not required.issubset(frame.columns):
            raise ValueError("Temporal split metadata is missing.")
        identifier_sets.append(
            set(zip(frame["source"], frame["window_index"]))
        )
    if any(
        identifier_sets[left] & identifier_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("Dataset leakage detected across temporal splits.")

def assert_disjoint_sources(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Raise when a source file occurs in more than one dataset split."""
    source_sets = []
    for frame in (train, val, test):
        if "source" not in frame.columns:
            raise ValueError("Source metadata is missing.")
        source_sets.append(set(frame["source"]))
    if any(
        source_sets[left] & source_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("Source-file leakage detected across dataset splits.")


def random_pool_test_split(
    df: pd.DataFrame,
    test_ratio: float,
    shuffle_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle rows (fixed seed, independent of the model seed) and cut off a test slice.

    Returns (pool, test); the pool is meant to be further divided into
    train/val via :func:`stratified_split`.
    """
    shuffled = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    test_size = int(len(shuffled) * test_ratio)
    return shuffled.iloc[test_size:].reset_index(drop=True), shuffled.iloc[:test_size].reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val per class, using an independent RNG."""
    rng = np.random.default_rng(seed)
    val_positions = []
    labels = df["label"].to_numpy()
    for lbl in np.unique(labels):
        pos = np.flatnonzero(labels == lbl)
        rng.shuffle(pos)
        n_val = int(round(len(pos) * val_ratio))
        val_positions.append(pos[:n_val])
    val_pos = np.sort(np.concatenate(val_positions))
    mask = np.zeros(len(df), dtype=bool)
    mask[val_pos] = True
    val_df = df.iloc[mask].reset_index(drop=True)
    train_df = df.iloc[~mask].reset_index(drop=True)
    return train_df, val_df


class TTADataset(Dataset):
    def __init__(self, dataset: Dataset, num_augmentations: int) -> None:
        """Create a version of the dataset that returns the same sample multiple times.

        This is useful for test-time augmentation (TTA).

        Args:
            dataset (Dataset): The dataset to be adapted for TTA.
            num_augmentations (int): The number of augmentations to apply.
        """
        super().__init__()
        self.dataset = dataset
        self.num_augmentations = num_augmentations

    def __len__(self) -> int:
        """Get the virtual length of the dataset."""
        return len(self.dataset) * self.num_augmentations

    def __getitem__(self, index) -> Any:
        """Get the item corresponding to idx // :attr:`self.num_augmentations`."""
        return self.dataset[index // self.num_augmentations]

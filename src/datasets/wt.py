# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
WT (Wind Turbine) Gearbox Dataset Module.

Dataset structure:
- 5 fault categories: broken, healthy, missing_tooth, root_crack, wear
- Each category has files with naming: {FaultPrefix}{Assembly}_{Speed}.MAT
    - Assembly: 1 or 2 (two different assemblies/拆装)
    - Speed: 20, 25, 30, 35, 40, 45, 50, 55 Hz
- Each MAT file contains 'Data' with shape (N, 4) - 4 channels
- Sample rate: 48000 Hz

Data Split Strategy:
================================================================================
- ID (In-Distribution): 3 fault types (broken, healthy, missing_tooth)
                        Assembly 1, speeds 30-40 Hz
- OOD (Out-of-Distribution): 2 fault types (root_crack, wear)
                             Assembly 1, speeds 30-40 Hz
- Shift: Same 3 ID fault types and Assembly 1, but unseen speeds
         (20, 25, 50, 55 Hz)

Keeping every split on Assembly 1 isolates speed shift from installation changes.
"""
import hashlib
import sys, warnings

warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.base_dataset import dataset
from src.datasets.noise import NoisyEvaluationMixin
from src.datasets.transforms import build_transforms
from src.datasets.cache import load_cached_array
from src.datasets.utils import (
    assert_disjoint_temporal_splits,
    concatenate_temporal_splits,
    temporal_window_split,
    subsample_uniform,
    random_pool_test_split,
    stratified_split,
)
from typing import Literal

signal_size = 1024
TEST_RATIO = 0.2
SPLIT_GAP_WINDOWS = 1
MAX_ID_WINDOWS_PER_FILE = 500
MAX_OOD_WINDOWS_PER_FILE = 100
MAX_SHIFT_WINDOWS_PER_FILE = 100

# ============================================================================
# Fault category definitions
# ============================================================================
# ID fault categories (used for training and testing)
ID_FAULT_CATEGORIES = {
    "broken": "B",
    "healthy": "N",
    "missing_tooth": "M",
}

# OOD fault categories (unseen during training)
OOD_FAULT_CATEGORIES = {
    "root_crack": "R",
    "wear": "W",
}

# ============================================================================
# File list generation
# ============================================================================
def generate_file_list(fault_categories: dict, assembly: int, speeds: list):
    """Generate file tuples (folder, filename, label) for given parameters."""
    files = []
    for label, (folder, prefix) in enumerate(fault_categories.items()):
        for speed in speeds:
            filename = f"{prefix}{assembly}_{speed}.MAT"
            files.append((folder, filename, label))
    return files


# ID: 3 fault types, Assembly 1, speeds 30-40 Hz
ID_FILES = generate_file_list(ID_FAULT_CATEGORIES, assembly=1, speeds=[30, 35, 40])

# Shift: Same ID fault types and assembly, unseen speeds
SHIFT_FILES = generate_file_list(
    ID_FAULT_CATEGORIES, assembly=1, speeds=[20, 25, 50, 55]
)

# OOD: Unseen fault types under the same assembly and ID speeds
OOD_FILES = generate_file_list(
    OOD_FAULT_CATEGORIES, assembly=1, speeds=[30, 35, 40]
)
# OOD 拆成两档，用于在集合过大时缩减评估样本数：
#   near —— root_crack (裂纹类故障)
#   far  —— wear (磨损类故障)
# 两者是不同故障类型而非严重度分层，划分依据类别而非难度。
OOD_FILES_NEAR = generate_file_list(
    {"root_crack": OOD_FAULT_CATEGORIES["root_crack"]}, assembly=1, speeds=[30, 35, 40]
)
OOD_FILES_FAR = generate_file_list(
    {"wear": OOD_FAULT_CATEGORIES["wear"]}, assembly=1, speeds=[30, 35, 40]
)

ID_LABELS = list(range(len(ID_FAULT_CATEGORIES)))  # 0~2 (3 classes)
SHIFT_LABELS = ID_LABELS  # Shift has the same 3 classes as ID
OOD_LABEL = -1

# Channel to use (0-3, using channel 0 by default)
DEFAULT_CHANNEL = 0


# ============================================================================
# Data loading functions
# ============================================================================
def load_signal_mat(filepath: str, channel: int = DEFAULT_CHANNEL) -> np.ndarray:
    """Load one MAT channel through the shared disk cache."""
    def parse(source: Path) -> np.ndarray:
        mat = sio.loadmat(source)
        data = mat["Data"]
        return data[:, channel].reshape(-1, 1)

    return load_cached_array(
        filepath,
        "wt",
        parse,
        parameters={"format": "mat", "field": "Data", "channel": channel},
    )


def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    """Slice the signal into non-overlapping windows."""
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]

    while end <= max_len:
        data.append(arr[start:end])
        labels.append(label)
        start += win
        end += win

    return data, labels


def data_load(
    filepath: str,
    label: int,
    channel: int = DEFAULT_CHANNEL,
    max_windows: int | None = None,
):
    """Load data from a MAT file and slice into windows."""
    arr = load_signal_mat(filepath, channel)
    data, labels = slice_windows(arr, label, win=signal_size)
    return subsample_uniform(data, labels, max_windows)


def build_df_from_files(
    root: str | Path,
    file_list: list,
    max_windows_per_file: int | None = None,
):
    """Build a DataFrame from a list of file tuples."""
    root = Path(root)
    all_data, all_labels, all_sources = [], [], []

    for folder, filename, label in file_list:
        filepath = root / folder / filename
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        d, l = data_load(
            str(filepath), label, max_windows=max_windows_per_file
        )
        all_data += d
        all_labels += l
        all_sources += [folder] * len(d)

    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


# ============================================================================
# DataModule class
# ============================================================================
def build_temporal_id_splits(
    root: str | Path,
    val_ratio: float,
    max_windows_per_file: int | None = MAX_ID_WINDOWS_PER_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every WT ID recording chronologically before concatenation."""
    root = Path(root)
    splits = []
    for folder, filename, label in ID_FILES:
        filepath = root / folder / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required WT ID file not found: {filepath}")
        data, labels = data_load(
            str(filepath), label, max_windows=max_windows_per_file
        )
        splits.append(
            temporal_window_split(
                data, labels, f"{folder}/{filename}", val_ratio,
                TEST_RATIO, SPLIT_GAP_WINDOWS,
            )
        )
    result = concatenate_temporal_splits(splits)
    assert_disjoint_temporal_splits(*result)
    return result


class WTDataModule(NoisyEvaluationMixin, TUDataModule):
    num_classes = 3  # ID fault types only
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["wt_ood"]

    def __init__(
            self,
            root: str | Path,
            batch_size: int,
            eval_batch_size: int | None = None,
            eval_ood: bool = False,
            eval_shift: bool = False,
            num_tta: int = 1,
            val_split: float | None = None,
            postprocess_set: Literal["val", "test"] = "val",
            num_workers: int = 1,
            train_transform: nn.Module | None = None,
            test_transform: nn.Module | None = None,
            ood_transform: nn.Module | None = None,
            normalize_type: str = "-1-1",
            pin_memory: bool = True,
            persistent_workers: bool = True,
            eval_noise: bool = False,
            noise_configs: list[tuple[str, int]] | None = None,
            split_seed: int = 12345,
            max_id_windows_per_file: int | None = MAX_ID_WINDOWS_PER_FILE,
            max_ood_windows_per_file: int | None = MAX_OOD_WINDOWS_PER_FILE,
            max_shift_windows_per_file: int | None = MAX_SHIFT_WINDOWS_PER_FILE,
            split_mode: Literal["random", "temporal"] = "temporal",
            ood_subset: Literal["all", "near", "far"] = "all",
    ) -> None:

        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_tta=num_tta,
            postprocess_set=postprocess_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.eval_noise = eval_noise
        self.noise_configs = noise_configs or [
            (noise_type, severity)
            for noise_type in self.noise_params
            for severity in range(1, 6)
        ]
        self.split_seed = split_seed
        self.normalize_type = normalize_type

        self.train_transform = build_transforms("train", normalize=self.normalize_type)
        self._split_done = False
        self.val_transform = build_transforms("val", normalize=self.normalize_type)
        self.max_id_windows_per_file = max_id_windows_per_file
        self.max_ood_windows_per_file = max_ood_windows_per_file
        self.max_shift_windows_per_file = max_shift_windows_per_file
        self.split_mode = split_mode
        self.ood_subset = ood_subset
        self.test_transform = build_transforms("val", normalize=self.normalize_type)
        self.ood_transform = build_transforms("val", normalize=self.normalize_type)

    def _ood_file_list(self) -> list:
        if self.ood_subset == "near":
            return OOD_FILES_NEAR
        if self.ood_subset == "far":
            return OOD_FILES_FAR
        return OOD_FILES

    def _build_splits(self) -> None:
        if self._split_done:
            return
        if not self.val_split:
            raise ValueError(
                "val_split must be positive to keep validation and test sets separate."
            )
        root = Path(self.root)
        if self.split_mode == "temporal":
            self.train_df, self.val_df, self.test_df = build_temporal_id_splits(
                root, self.val_split, self.max_id_windows_per_file
            )
        else:
            id_df = build_df_from_files(root, ID_FILES, self.max_id_windows_per_file)
            # Test cut uses a fixed shuffle seed, independent of the model seed.
            pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed
            )
        if self.eval_ood:
            self.ood_df = build_df_from_files(
                root, self._ood_file_list(), self.max_ood_windows_per_file
            )
            self.ood_df["label"] = OOD_LABEL
        if self.eval_shift:
            self.shift_df = build_df_from_files(
                root, SHIFT_FILES, self.max_shift_windows_per_file
            )
        self._split_done = True

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if getattr(self, "_noisy_mode", False):
            return
        self._build_splits()
        self.train = dataset(
            list_data=self.train_df, transform=self.train_transform
        )
        self.val = dataset(
            list_data=self.val_df, transform=self.test_transform
        )
        if stage in ("test", None):
            self.test = dataset(
                list_data=self.test_df, transform=self.test_transform
            )
        if self.eval_ood:
            self.ood = dataset(
                list_data=self.ood_df, transform=self.ood_transform
            )
        if self.eval_shift:
            self.shift = dataset(
                list_data=self.shift_df, transform=self.test_transform
            )

    def split_summary(self) -> str:
        """打印各划分的样本数与每类分布，用于确认划分不随 seed 变化。"""
        self._build_splits()
        lines = [f"split_mode={self.split_mode}"]
        for name, df in [("train", self.train_df), ("val", self.val_df),
                         ("test", self.test_df)]:
            counts = df["label"].value_counts().sort_index().to_dict()
            lines.append(f"{name:6s} n={len(df):5d}  per-class={counts}")
        if self.eval_ood:
            lines.append(f"{'ood':6s} n={len(self.ood_df):5d}  "
                         f"subset={self.ood_subset}  "
                         f"max_per_file={self.max_ood_windows_per_file}")
        if self.eval_shift:
            lines.append(f"{'shift':6s} n={len(self.shift_df):5d}")
        return "\n".join(lines)
    

    def test_dataloader(self) -> list[DataLoader]:
        dataloaders = [
            self._data_loader(self.get_test_set(), training=False, shuffle=False)
        ]
        if self.eval_ood:
            dataloaders.append(
                self._data_loader(self.get_ood_set(), training=False, shuffle=False)
            )
        if self.eval_shift:
            dataloaders.append(
                self._data_loader(self.get_shift_set(), training=False, shuffle=False)
            )
        return dataloaders



if __name__ == "__main__":
    import torch

    for mode in ["random", "temporal"]:
        print(f"\n{'=' * 60}\n  split_mode = {mode}\n{'=' * 60}")
        fingerprints = []
        for s in [0, 1, 2]:
            torch.manual_seed(s)
            np.random.seed(s)
            dm = WTDataModule(root="/mnt/d/Data/Machine/WT", batch_size=64,
                               val_split=0.2, eval_ood=True, split_mode=mode)
            dm.setup("test")
            print(f"\n--- global seed {s} ---")
            print(dm.split_summary())
            fp = hashlib.md5(
                str(dm.val_df["label"].tolist()
                    + [float(a.sum()) for a in dm.val_df["data"]]).encode()
            ).hexdigest()[:12]
            fingerprints.append(fp)
            print("val fingerprint:", fp)

        print("\n划分是否稳定:",
              "是" if len(set(fingerprints)) == 1 else "否 —— 仍受全局随机状态影响")
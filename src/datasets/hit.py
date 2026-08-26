# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
HIT Aerospace Engine Intershaft Bearing Dataset
HIT 航空发动机轴间轴承数据集

Dataset from Harbin Institute of Technology dual-rotor test platform.
数据来自哈尔滨工业大学双转子试验平台。

Data Structure:
- Sampling frequency: 25 kHz
- Sample length: 20480 points
- Channels: 6 (2 displacement + 4 acceleration)
- Format: .npy tensor N×6×20480

Subsets:
- data1: Normal bearing (28 conditions, 504 samples)
- data2: Normal bearing (25 conditions, 450 samples) 
- data3: Inner fault bearing A (28 conditions, 504 samples)
- data4: Inner fault bearing B (28 conditions, 504 samples)
- data5: Outer fault bearing (25 conditions, 450 samples)
"""

import hashlib
import sys, warnings

warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import numpy as np
from src.datasets.cache import load_cached_array
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.base_dataset import dataset
from src.datasets.noise import NoisyEvaluationMixin
from src.datasets.transforms import build_transforms
from src.datasets.utils import (
    assert_disjoint_temporal_splits,
    concatenate_temporal_splits,
    temporal_window_split,
    subsample_uniform,
    random_pool_test_split,
    stratified_split,
)
from typing import Literal, List, Optional, Union

# Default signal size (can be overridden)
signal_size = 1024
TEST_RATIO = 0.2
SPLIT_GAP_WINDOWS = 1

# ID data: Normal + Inner fault
ID_FILES = ["data1.npy", "data3.npy"]
ID_LABELS = [0, 1]  # 0: Normal, 1: Inner fault

# Shift data: Different conditions/crack lengths
SHIFT_FILES = ["data2.npy", "data4.npy"]
SHIFT_LABELS = [0, 1]  # Same label mapping

# OOD data: Outer fault (unseen fault type)
OOD_FILES = ["data5.npy"]
OOD_LABEL = -1
# HIT only ships a single OOD recording, so there is no near/far grouping to
# make: "near"/"far"/"all" all resolve to the same file list.
OOD_FILES_NEAR = OOD_FILES
OOD_FILES_FAR = OOD_FILES


def load_npy_data(
    filepath: Path,
    label: int,
    signal_size: int = 1024,
    channels: Optional[List[int]] = None,
    max_windows: int | None = None,
) -> tuple:
    """
    Load .npy file and slice samples.
    加载 .npy 文件并切片样本。
    
    Args:
        filepath: Path to .npy file
        label: Label for all samples in this file
        signal_size: Length of signal to extract (default 1024)
        channels: List of channel indices to use (0-5). 
                  None = all 6 channels.
                  Example: [0] for single channel, [0,1,2] for first 3 channels
    
    Returns:
        (data_list, label_list)
    """
    if channels is None:
        channels = list(range(6))
    selected_channels = list(channels)

    def parse(source: Path) -> np.ndarray:
        raw = np.load(source, allow_pickle=False, mmap_mode="r")
        selected = raw[:, selected_channels, :signal_size]
        return selected.transpose(0, 2, 1)

    arr = load_cached_array(
        filepath,
        "hit",
        parse,
        parameters={
            "format": "npy",
            "channels": selected_channels,
            "signal_size": signal_size,
            "layout": "NLC",
        },
    )
    data = [sample for sample in arr]
    labels = [label] * len(data)

    return subsample_uniform(data, labels, max_windows)


def build_df_from_files(
    root: Path,
    file_list: List[str],
    label_list: List[int],
    signal_size: int = 1024,
    channels: Optional[List[int]] = None,
    max_windows_per_file: int | None = None,
) -> pd.DataFrame:
    """
    Build DataFrame from multiple .npy files.
    从多个 .npy 文件构建 DataFrame。
    """
    all_data, all_labels, all_sources = [], [], []

    for fname, lbl in zip(file_list, label_list):
        path = root / fname
        if not path.exists():
            print(f"Warning: {path} not found, skipping...")
            continue
        d, l = load_npy_data(path, lbl, signal_size, channels, max_windows_per_file)
        all_data += d
        all_labels += l
        all_sources += [fname] * len(d)

    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


def build_temporal_id_splits(
    root: Path,
    val_ratio: float,
    signal_size: int,
    channels: Optional[List[int]],
    max_windows_per_file: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every HIT ID array chronologically before concatenation."""
    splits = []
    for filename, label in zip(ID_FILES, ID_LABELS):
        filepath = root / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required HIT ID file not found: {filepath}")
        data, labels = load_npy_data(
            filepath, label, signal_size, channels, max_windows_per_file
        )
        splits.append(
            temporal_window_split(
                data, labels, filename, val_ratio,
                TEST_RATIO, SPLIT_GAP_WINDOWS,
            )
        )
    result = concatenate_temporal_splits(splits)
    assert_disjoint_temporal_splits(*result)
    return result


class HITDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    HIT Aerospace Engine Intershaft Bearing DataModule.
    HIT 航空发动机轴间轴承数据模块。
    
    Args:
        root: Path to data directory (./data/hit)
        batch_size: Batch size for training
        channels: Channel indices to use. Options:
            - None: All 6 channels (default)
            - [0]: Single channel (channel 0)
            - [0, 1]: First 2 channels
            - [0, 1, 2, 3, 4, 5]: All channels explicitly
        signal_size: Length of signal per sample (default 1024)
        eval_ood: Whether to evaluate OOD detection
        eval_shift: Whether to evaluate domain shift
    
    Example:
        # Use all 6 channels
        dm = HITDataModule(root="./data/hit", batch_size=32)
        
        # Use single channel
        dm = HITDataModule(root="./data/hit", batch_size=32, channels=[0])
        
        # Use first 3 channels
        dm = HITDataModule(root="./data/hit", batch_size=32, channels=[0, 1, 2])
    """
    
    training_task = "classification"
    ood_datasets = ["hit_outer_fault_ood"]

    def __init__(
            self,
            root: str | Path,
            batch_size: int,
            channels: Optional[List[int]] = None,
            signal_size: int = 1024,
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
            split_mode: Literal["random", "temporal"] = "temporal",
            ood_subset: Literal["all", "near", "far"] = "all",
            max_id_windows_per_file: int | None = None,
            max_ood_windows_per_file: int | None = None,
            max_shift_windows_per_file: int | None = None,
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

        # Channel selection
        self.channels = channels if channels is not None else list(range(6))
        self.signal_size = signal_size
        
        # Set class attributes based on channel selection
        self.num_channels = len(self.channels)
        self.input_shape = (self.num_channels, self.signal_size)
        self.num_classes = 2  # Normal, Inner fault
        
        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.eval_noise = eval_noise
        self.noise_configs = noise_configs or [
            (noise_type, severity)
            for noise_type in self.noise_params
            for severity in range(1, 6)
        ]
        self.split_seed = split_seed
        self.split_mode = split_mode
        self.ood_subset = ood_subset
        self.max_id_windows_per_file = max_id_windows_per_file
        self.max_ood_windows_per_file = max_ood_windows_per_file
        self.max_shift_windows_per_file = max_shift_windows_per_file
        self.normalize_type = normalize_type

        self.train_transform = build_transforms("train", normalize=self.normalize_type)
        self._split_done = False
        self.val_transform = build_transforms("val", normalize=self.normalize_type)
        self.test_transform = build_transforms("val", normalize=self.normalize_type)
        self.ood_transform = build_transforms("val", normalize=self.normalize_type)

    def _ood_file_list(self) -> List[str]:
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
                root, self.val_split, self.signal_size, self.channels,
                self.max_id_windows_per_file,
            )
        else:
            id_df = build_df_from_files(
                root, ID_FILES, ID_LABELS, self.signal_size, self.channels,
                self.max_id_windows_per_file,
            )
            # Test cut uses a fixed shuffle seed, independent of the model seed.
            pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed
            )
        if self.eval_ood:
            self.ood_df = build_df_from_files(
                root, self._ood_file_list(), [OOD_LABEL],
                self.signal_size, self.channels, self.max_ood_windows_per_file,
            )
        if self.eval_shift:
            self.shift_df = build_df_from_files(
                root, SHIFT_FILES, SHIFT_LABELS,
                self.signal_size, self.channels, self.max_shift_windows_per_file,
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
            dm = HITDataModule(root="/mnt/d/Data/Machine/HIT", batch_size=64,
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
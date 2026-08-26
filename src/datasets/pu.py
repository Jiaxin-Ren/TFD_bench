# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
PU Bearing Dataset / 帕德博恩大学轴承数据集

Paderborn University Bearing Data Center Dataset.
64kHz sampling rate with multiple fault types and operating conditions.

Dataset structure:
    pu/
    └── PU-dataset-main/
        ├── N09_M07_F10/  (转速900rpm, 负载矩0.7Nm, 径向力10N)
        │   ├── N09_M07_F10_K001_1.mat
        │   ├── N09_M07_F10_KA04_1.mat
        │   └── ...
        ├── N15_M01_F10/  (转速1500rpm, 负载矩0.1Nm, 径向力10N)
        ├── N15_M07_F04/  (转速1500rpm, 负载矩0.7Nm, 径向力4N)
        └── N15_M07_F10/  (转速1500rpm, 负载矩0.7Nm, 径向力10N)

Reference: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter

Classes / 类别:
    ID (13 classes): KA04, KA15, KA16, KA22, KA30, KB23, KB24, KB27, KI04, KI16, KI17, KI18, KI21
    OOD: K001-K006 (正常健康状态 - 作为OOD)
    Shift: 不同工况下的相同类别 (例如从N15_M07_F10到N09_M07_F10)
    
Fault Types / 故障类型:
    - K001-K006: 健康轴承 (6 types)
    - KA系列: 外圈故障 (Outer Race - Außenring)
    - KI系列: 内圈故障 (Inner Race - Innenring)
    - KB系列: 滚动体故障 (Rolling Element - Rollkörper)
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
from typing import Literal, List, Optional

signal_size = 1024

# =============================================================================
# Working Conditions / 工况定义
# =============================================================================

# ID工况: N15_M07_F10 (转速1500rpm, 负载矩0.7Nm, 径向力10N) - 最常用工况
ID_CONDITION = "N15_M07_F10"

# Shift工况: 不同的操作条件
TEST_RATIO = 0.2
SPLIT_GAP_WINDOWS = 1
SHIFT_CONDITIONS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]

# 所有工况
ALL_CONDITIONS = [ID_CONDITION] + SHIFT_CONDITIONS

# =============================================================================
# Fault Type Mapping / 故障类型映射
# =============================================================================

# ID类别: 外圈(KA) + 内圈(KI) + 滚动体(KB) 故障类别
ID_CLASSES = [
    "K001",  # 健康轴承 (1)
    "KA04", "KA15", "KA16", "KA22", "KA30",  # 外圈故障 (5)
    "KB23", "KB24", "KB27",                    # 滚动体故障 (3)
]
ID_LABELS = {cls: i for i, cls in enumerate(ID_CLASSES)}
NUM_ID_CLASSES = len(ID_CLASSES)

# OOD类别: 健康轴承 (训练时未见)
OOD_CLASSES = ["KI04", "KI16", "KI17", "KI18", "KI21"]  # 内圈故障 (5)
OOD_LABEL = -1
OOD_LABELS = {cls: OOD_LABEL for cls in OOD_CLASSES}

# PU 的 K-系列损伤编号没有公开的严重度排序，因此 near/far 只是按列表
# 位置对半拆分，用于在 OOD 集合过大时降低样本数，而不是严格的难度分层。
OOD_CLASSES_NEAR = ["KI04", "KI16"]
OOD_CLASSES_FAR = ["KI17", "KI18", "KI21"]

# 所有轴承类别
ALL_BEARING_CLASSES = ID_CLASSES + OOD_CLASSES


def _read_mat_file(filepath: str) -> np.ndarray:
    """
    Load vibration signal from PU .mat file.
    从PU .mat文件加载振动信号。
    
    PU dataset MAT file structure:
        data[main_key][0,0]['Y'][0, channel_idx]['Data']
    
    The vibration data is in the channel named 'vibration_1'.
    """
    mat_data = sio.loadmat(filepath)
    
    # Get the main key (e.g., 'N15_M07_F10_KA04_1')
    main_key = None
    for key in mat_data.keys():
        if not key.startswith('_'):
            main_key = key
            break
    
    if main_key is None:
        raise ValueError(f"Could not find main key in {filepath}")
    
    try:
        # Navigate the nested structure: data[main_key][0,0]['Y']
        inner = mat_data[main_key][0, 0]
        Y = inner['Y']  # Shape: (1, num_channels)
        
        # Search for the vibration channel
        for i in range(Y.shape[1]):
            channel = Y[0, i]
            name = channel['Name'].flatten()[0] if channel['Name'].size > 0 else ''
            
            if 'vibration' in name.lower():
                vibration_data = channel['Data'].flatten()
                return vibration_data
        
        raise ValueError(f"Could not find vibration channel in {filepath}")
        
    except Exception as e:
        # Fallback: try the old method for backwards compatibility
        for key in mat_data.keys():
            if key.startswith('_'):
                continue
            arr = mat_data[key]
            if isinstance(arr, np.ndarray):
                if arr.dtype == np.object_:
                    try:
                        inner = arr[0, 0]
                        for i in range(len(inner)):
                            if isinstance(inner[i], np.ndarray) and inner[i].dtype in [np.float64, np.float32]:
                                if inner[i].size > 1000:
                                    return inner[i].flatten()
                    except (IndexError, TypeError):
                        pass
                elif arr.size > 1000:
                    return arr.flatten()
        
        raise ValueError(f"Could not find vibration data in {filepath}: {e}")

def load_mat_file(filepath: str) -> np.ndarray:
    """Load the parsed vibration channel through the shared disk cache."""
    return load_cached_array(
        filepath,
        "pu",
        lambda source: _read_mat_file(str(source)),
        parameters={"format": "mat", "channel": "vibration"},
    )



def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    """Slice signal into fixed-length windows / 将信号切分为固定长度窗口"""
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]

    while end <= max_len:
        segment = arr[start:end].reshape(-1, 1)  # Shape: (win, 1)
        data.append(segment)
        labels.append(label)
        start += win
        end += win

    return data, labels


def data_load(filepath: str, label: int, max_windows: int | None = None):
    """Load and slice data from a .mat file / 从.mat文件加载并切分数据"""
    arr = load_mat_file(filepath)
    data, labels = slice_windows(arr, label, win=signal_size)
    return subsample_uniform(data, labels, max_windows)


def get_file_path(root: Path, condition: str, bearing_class: str) -> Path:
    """
    Get the file path for a specific condition and bearing class.
    获取特定工况和轴承类别的文件路径。
    
    File naming convention: {condition}_{bearing_class}_1.mat
    """
    return root / "PU-dataset-main" / condition / f"{condition}_{bearing_class}_1.mat"


def build_df_from_files(
    root: Path,
    condition: str,
    classes: List[str],
    label_dict: dict,
    max_windows_per_file: int | None = None,
) -> pd.DataFrame:
    """
    Build DataFrame from files for a specific condition.
    从特定工况的文件构建DataFrame。
    """
    all_data, all_labels, all_sources = [], [], []

    for bearing_class in classes:
        filepath = get_file_path(root, condition, bearing_class)

        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        label = label_dict.get(bearing_class, 0)

        try:
            d, l = data_load(str(filepath), label, max_windows_per_file)
            all_data += d
            all_labels += l
            all_sources += [bearing_class] * len(d)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


def build_df_from_multiple_conditions(
    root: Path,
    conditions: List[str],
    classes: List[str],
    label_dict: dict,
    max_windows_per_file: int | None = None,
) -> pd.DataFrame:
    """
    Build DataFrame from files across multiple conditions.
    从多个工况的文件构建DataFrame。
    """
    dfs = []
    for condition in conditions:
        df = build_df_from_files(root, condition, classes, label_dict, max_windows_per_file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_temporal_id_splits(
    root: Path,
    val_ratio: float,
    max_windows_per_file: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every ID bearing recording chronologically before concatenation."""
    splits = []
    for bearing_class in ID_CLASSES:
        filepath = get_file_path(root, ID_CONDITION, bearing_class)
        if not filepath.exists():
            raise FileNotFoundError(f"Required PU ID file not found: {filepath}")
        label = ID_LABELS[bearing_class]
        data, labels = data_load(str(filepath), label, max_windows_per_file)
        splits.append(
            temporal_window_split(
                data, labels, f"{ID_CONDITION}/{filepath.name}",
                val_ratio, TEST_RATIO, SPLIT_GAP_WINDOWS,
            )
        )
    result = concatenate_temporal_splits(splits)
    assert_disjoint_temporal_splits(*result)
    return result

class PUDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    PU Bearing Dataset DataModule.
    帕德博恩大学轴承数据集 DataModule。
    
    ID: 13 fault classes (KA04, KA15, KA16, KA22, KA30, KB23, KB24, KB27, 
                          KI04, KI16, KI17, KI18, KI21) under N15_M07_F10 condition
    Shift: Same 13 classes under different operating conditions
    OOD: K001-K006 (Normal/Healthy bearings - unseen in training)
    """
    
    num_classes = NUM_ID_CLASSES  # 13 classes
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["pu_healthy_ood"]

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
            split_mode: Literal["random", "temporal"] = "temporal",
            ood_subset: Literal["all", "near", "far"] = "near",
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

    def _ood_class_list(self) -> List[str]:
        if self.ood_subset == "near":
            return OOD_CLASSES_NEAR
        if self.ood_subset == "far":
            return OOD_CLASSES_FAR
        return OOD_CLASSES

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
            id_df = build_df_from_files(
                root, ID_CONDITION, ID_CLASSES, ID_LABELS, self.max_id_windows_per_file
            )
            # Test cut uses a fixed shuffle seed, independent of the model seed.
            pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed
            )
        if self.eval_ood:
            classes = self._ood_class_list()
            self.ood_df = build_df_from_files(
                root, ID_CONDITION, classes, OOD_LABELS, self.max_ood_windows_per_file
            )
        if self.eval_shift:
            self.shift_df = build_df_from_multiple_conditions(
                root, SHIFT_CONDITIONS, ID_CLASSES, ID_LABELS,
                self.max_shift_windows_per_file,
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
            dm = PUDataModule(root="/mnt/d/Data/Machine/PU", batch_size=64,
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
# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
XJTU Gearbox Dataset / 西安交通大学齿轮箱数据集

XJTU Gearbox and Bearing Fault Diagnosis Dataset.
Includes bearing faults and planetary gear faults.

Dataset structure:
    xjtu/
    ├── 1ndBearing_ball/
    │   ├── Data_Chan1.txt
    │   └── Data_Chan2.txt
    ├── 1ndBearing_inner/
    ├── 1ndBearing_mix(inner+outer+ball)/
    ├── 1ndBearing_outer/
    ├── 2ndPlanetary_brokentooth/
    ├── 2ndPlanetary_missingtooth/
    ├── 2ndPlanetary_normalstate/
    ├── 2ndPlanetary_rootcracks/
    └── 2ndPlanetary_toothwear/

Classes / 类别:
    ID (9 classes): All fault types using Channel 1
    Shift: Same classes using Channel 2 (different sensor position)
    OOD: Can be configured (e.g., specific fault types unseen in training)

Fault Types / 故障类型:
    - 1ndBearing_*: 轴承故障 (ball, inner, outer, mix)
    - 2ndPlanetary_*: 行星齿轮故障 (brokentooth, missingtooth, normalstate, rootcracks, toothwear)
"""
import hashlib
import sys, warnings

warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from itertools import islice
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.cache import load_cached_array
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
from typing import Literal, List, Optional

signal_size = 1024

TEST_RATIO = 0.2
SPLIT_GAP_WINDOWS = 1
MAX_ID_WINDOWS_PER_FILE = 500
MAX_OOD_WINDOWS_PER_FILE = 800
MAX_SHIFT_WINDOWS_PER_FILE = 500
# =============================================================================
# Class Definitions / 类别定义
# =============================================================================

# All fault classes
ALL_CLASSES = [
    "1ndBearing_ball",
    "1ndBearing_inner",
    "1ndBearing_mix(inner+outer+ball)",
    "1ndBearing_outer",
    "2ndPlanetary_brokentooth",
    "2ndPlanetary_missingtooth",
    "2ndPlanetary_normalstate",
    "2ndPlanetary_rootcracks",
    "2ndPlanetary_toothwear",
]

# ID类别: 全部9类
ID_CLASSES = ALL_CLASSES.copy()
ID_LABELS = {cls: i for i, cls in enumerate(ID_CLASSES)}
NUM_ID_CLASSES = len(ID_CLASSES)

# OOD类别: 可以根据需要配置，这里使用mix类型作为OOD
# 因为mix类型是复合故障，更难识别
OOD_CLASSES = ["1ndBearing_mix(inner+outer+ball)"]
OOD_LABEL = -1
# XJTU only ships one OOD fault class, so there is no near/far grouping to
# make: "near"/"far"/"all" all resolve to the same class list.
OOD_CLASSES_NEAR = OOD_CLASSES
OOD_CLASSES_FAR = OOD_CLASSES

# ID类别(排除OOD)
ID_CLASSES_NO_OOD = [c for c in ID_CLASSES if c not in OOD_CLASSES]
ID_LABELS_NO_OOD = {cls: i for i, cls in enumerate(ID_CLASSES_NO_OOD)}

# Channel definitions
CHANNEL_ID = "Data_Chan1.txt"      # ID数据使用通道1
CHANNEL_SHIFT = "Data_Chan2.txt"   # Shift数据使用通道2

# Header lines to skip in txt files
HEADER_LINES = 14


def _read_signal_txt(filepath: str) -> np.ndarray:
    """
    Load vibration signal from XJTU txt file.
    从XJTU txt文件加载振动信号。
    
    The file has a header section followed by numerical data.
    """
    data = []
    with open(filepath, "r", errors="ignore") as f:
        for line in islice(f, HEADER_LINES, None):
            try:
                val = float(line.strip())
                data.append(val)
            except ValueError:
                continue
    
    return np.array(data).reshape(-1, 1)

def load_signal_txt(filepath: str) -> np.ndarray:
    """Load the parsed text signal through the shared disk cache."""
    return load_cached_array(
        filepath,
        "xjtu",
        lambda source: _read_signal_txt(str(source)),
        parameters={"format": "text", "skip_rows": HEADER_LINES},
    )



def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    """Slice signal into fixed-length windows / 将信号切分为固定长度窗口"""
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
    max_windows: int | None = None,
):
    """Load and slice data from a txt file / 从txt文件加载并切分数据"""
    arr = load_signal_txt(filepath)
    data, labels = slice_windows(arr, label, win=signal_size)
    return subsample_uniform(data, labels, max_windows)


def build_df_from_classes(
    root: Path, 
    classes: List[str], 
    label_dict: dict,
    channel: str = CHANNEL_ID,
    max_windows_per_file: int | None = None,
) -> pd.DataFrame:
    """
    Build DataFrame from class folders.
    从类别文件夹构建DataFrame。
    """
    all_data, all_labels, all_sources = [], [], []
    
    for cls in classes:
        filepath = root / cls / channel
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        label = label_dict.get(cls, 0)
        
        try:
            d, l = data_load(
                str(filepath), label, max_windows=max_windows_per_file
            )
            all_data += d
            all_labels += l
            all_sources += [cls] * len(d)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


def build_temporal_id_splits(
    root: Path,
    val_ratio: float,
    max_windows_per_file: int | None = MAX_ID_WINDOWS_PER_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every XJTU ID recording chronologically before concatenation."""
    splits = []
    for fault_class in ID_CLASSES_NO_OOD:
        filepath = root / fault_class / CHANNEL_ID
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required XJTU ID file not found: {filepath}"
            )
        label = ID_LABELS_NO_OOD[fault_class]
        data, labels = data_load(
            str(filepath), label, max_windows=max_windows_per_file
        )
        splits.append(
            temporal_window_split(
                data, labels, f"{fault_class}/{CHANNEL_ID}", val_ratio,
                TEST_RATIO, SPLIT_GAP_WINDOWS,
            )
        )
    result = concatenate_temporal_splits(splits)
    assert_disjoint_temporal_splits(*result)
    return result


class XJTUDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    XJTU Gearbox Dataset DataModule.
    西安交通大学齿轮箱数据集 DataModule。
    
    ID: 8 fault classes (excluding mix) using Channel 1
    Shift: Same 8 classes using Channel 2 (different sensor)
    OOD: 1ndBearing_mix (compound fault - unseen in training)
    """
    
    num_classes = len(ID_CLASSES_NO_OOD)  # 8 classes (excluding OOD)
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["xjtu_mix_fault_ood"]

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

    def _ood_class_list(self) -> list:
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
            id_df = build_df_from_classes(
                root, ID_CLASSES_NO_OOD, ID_LABELS_NO_OOD, CHANNEL_ID,
                self.max_id_windows_per_file,
            )
            # Test cut uses a fixed shuffle seed, independent of the model seed.
            pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed
            )
        if self.eval_ood:
            classes = self._ood_class_list()
            ood_labels = {cls: OOD_LABEL for cls in classes}
            self.ood_df = build_df_from_classes(
                root, classes, ood_labels, CHANNEL_ID,
                self.max_ood_windows_per_file,
            )
        if self.eval_shift:
            self.shift_df = build_df_from_classes(
                root, ID_CLASSES_NO_OOD, ID_LABELS_NO_OOD, CHANNEL_SHIFT,
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
            dm = XJTUDataModule(root="/mnt/d/Data/Machine/XJTU", batch_size=64,
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
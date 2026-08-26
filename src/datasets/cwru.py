# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
CWRU Bearing Dataset / 凯斯西储大学轴承数据集

Case Western Reserve University Bearing Fault Dataset.
12kHz Drive End bearing data with multiple fault types and load conditions.

Dataset structure:
    CWRU/
    └── 12k/
        ├── 97.mat   (Normal 0HP)
        ├── 105.mat  (IR007 0HP)
        ├── 118.mat  (B007 0HP)
        ├── 130.mat  (OR007@6 0HP)
        └── ...

Reference: https://engineering.case.edu/bearingdatacenter

Classes / 类别:
    ID (7 classes): Normal, IR007, IR014, IR021, B007, B014, B021
    OOD (Outer Race faults): OR007, OR014, OR021
    Shift: Same classes but different load (3HP instead of 0HP)
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
import torch
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
from typing import Literal
import scipy.io as sio


signal_size = 1024

# =============================================================================
# File Number to Fault Type Mapping (12kHz Drive End)
from src.datasets.cache import load_cached_array
# 文件编号与故障类型对应关系 (12kHz 驱动端)
# =============================================================================

# Format: file_number -> (fault_type, load_hp)
FILE_MAPPING = {
    # Normal / 正常
    97: ("Normal", 0),
    98: ("Normal", 1),
    99: ("Normal", 2),
    100: ("Normal", 3),
    
    # Inner Race 0.007" / 内圈故障 0.007"
    105: ("IR007", 0),
    106: ("IR007", 1),
    107: ("IR007", 2),
    108: ("IR007", 3),
    
    # Inner Race 0.014" / 内圈故障 0.014"
    169: ("IR014", 0),
    170: ("IR014", 1),
    171: ("IR014", 2),
    172: ("IR014", 3),
    
    # Inner Race 0.021" / 内圈故障 0.021"
    209: ("IR021", 0),
    210: ("IR021", 1),
    211: ("IR021", 2),
    212: ("IR021", 3),
    
    # Ball 0.007" / 滚动体故障 0.007"
    118: ("B007", 0),
    119: ("B007", 1),
    120: ("B007", 2),
    121: ("B007", 3),
    
    # Ball 0.014" / 滚动体故障 0.014"
    185: ("B014", 0),
    186: ("B014", 1),
    187: ("B014", 2),
    188: ("B014", 3),
    
    # Ball 0.021" / 滚动体故障 0.021"
    222: ("B021", 0),
    223: ("B021", 1),
    224: ("B021", 2),
    225: ("B021", 3),
    
    # Outer Race 0.007" @6 o'clock / 外圈故障 0.007" 6点钟位置
    130: ("OR007", 0),
    131: ("OR007", 1),
    132: ("OR007", 2),
    133: ("OR007", 3),
    
    # Outer Race 0.014" @6 o'clock / 外圈故障 0.014" 6点钟位置
    197: ("OR014", 0),
    198: ("OR014", 1),
    199: ("OR014", 2),
    200: ("OR014", 3),
    
    # Outer Race 0.021" @6 o'clock / 外圈故障 0.021" 6点钟位置
    234: ("OR021", 0),
    235: ("OR021", 1),
    236: ("OR021", 2),
    237: ("OR021", 3),
}

# ID类别 (用于训练): Normal + Inner Race + Ball
ID_CLASSES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021"]
ID_LABELS = {cls: i for i, cls in enumerate(ID_CLASSES)}

# OOD类别: Outer Race (训练时未见)
OOD_CLASSES = ["OR007", "OR014", "OR021"]
OOD_LABEL = -1
OOD_LABELS = {cls: OOD_LABEL for cls in OOD_CLASSES}

# OOD 按故障尺寸(defect diameter)拆成两档：
#   near —— 0.007" 小尺寸故障，幅值与 ID 中最小尺寸故障(IR007/B007)接近
#   far  —— 0.014"/0.021" 较大尺寸故障，特征更明显、更容易与 ID 区分
OOD_CLASSES_NEAR = ["OR007"]
OOD_CLASSES_FAR = ["OR014", "OR021"]

# 根据负载划分
ID_LOAD = 0    # ID数据使用0HP负载
SHIFT_LOAD = 3  # Shift数据使用3HP负载
TEST_RATIO = 0.2
SPLIT_GAP_WINDOWS = 1


def get_files_for_condition(load_hp: int, classes: list) -> list:
    """Get file numbers for given load and classes / 获取指定负载和类别的文件编号"""
    files = []
    for file_num, (fault_type, load) in FILE_MAPPING.items():
        if load == load_hp and fault_type in classes:
            files.append(file_num)
    return sorted(files)


# ID Files (0HP load) / ID文件 (0HP负载)
ID_FILES = get_files_for_condition(ID_LOAD, ID_CLASSES)
# Shift Files (3HP load) / Shift文件 (3HP负载)  
SHIFT_FILES = get_files_for_condition(SHIFT_LOAD, ID_CLASSES)
# OOD Files (all loads) / OOD文件 (所有负载)
OOD_FILES = get_files_for_condition(0, OOD_CLASSES) + get_files_for_condition(3, OOD_CLASSES)


def ood_files_for_classes(classes: list) -> list:
    """OOD file numbers (both loads) restricted to the given fault classes."""
    return get_files_for_condition(0, classes) + get_files_for_condition(3, classes)


def _read_mat_file(filepath: str) -> np.ndarray:
    """
    Load vibration signal from .mat file.
    从.mat文件加载振动信号。
    
    CWRU .mat files contain drive end (DE) acceleration data.
    Variable names vary, so we search for the one containing 'DE_time'.
    """
    mat_data = sio.loadmat(filepath)
    
    # Find the drive end time series data
    # 查找驱动端时间序列数据
    for key in mat_data.keys():
        if 'DE_time' in key:
            return mat_data[key].flatten()
    
    # Fallback: try to find any acceleration data
    for key in mat_data.keys():
        if not key.startswith('_') and isinstance(mat_data[key], np.ndarray):
            arr = mat_data[key]
            if arr.ndim <= 2 and arr.size > 1000:
                return arr.flatten()
    
    raise ValueError(f"Could not find vibration data in {filepath}")

def load_mat_file(filepath: str) -> np.ndarray:
    """Load the drive-end signal through the shared disk cache."""
    return load_cached_array(
        filepath,
        "cwru",
        lambda source: _read_mat_file(str(source)),
        parameters={"format": "mat", "signal": "DE_time"},
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


def build_df_from_files(
    root: str | Path,
    file_list: list,
    class_list: list,
    label_dict: dict,
    max_windows_per_file: int | None = None,
):
    """
    Build DataFrame from file list.
    从文件列表构建DataFrame。

    Args:
        root: Path to data directory
        file_list: List of file numbers
        class_list: List of class names to include
        label_dict: Mapping from class name to label
        max_windows_per_file: Optional cap on windows kept per file
    """
    root = Path(root)
    all_data, all_labels, all_sources = [], [], []

    for file_num in file_list:
        if file_num not in FILE_MAPPING:
            continue

        fault_type, _ = FILE_MAPPING[file_num]
        if fault_type not in class_list:
            continue

        label = label_dict[fault_type]
        filepath = root / f"{file_num}.mat"

        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        try:
            d, l = data_load(str(filepath), label, max_windows_per_file)
            all_data += d
            all_labels += l
            all_sources += [fault_type] * len(d)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


def build_temporal_id_splits(
    root: str | Path,
    val_ratio: float,
    max_windows_per_file: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every CWRU ID recording chronologically before concatenation."""
    root = Path(root)
    splits = []
    for file_num in ID_FILES:
        fault_type, _ = FILE_MAPPING[file_num]
        filepath = root / f"{file_num}.mat"
        if not filepath.exists():
            raise FileNotFoundError(f"Required CWRU ID file not found: {filepath}")
        label = ID_LABELS[fault_type]
        data, labels = data_load(str(filepath), label, max_windows_per_file)
        splits.append(
            temporal_window_split(
                data, labels, filepath.name, val_ratio,
                TEST_RATIO, SPLIT_GAP_WINDOWS,
            )
        )
    result = concatenate_temporal_splits(splits)
    assert_disjoint_temporal_splits(*result)
    return result


class CWRUDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    CWRU Bearing Dataset DataModule.
    凯斯西储大学轴承数据集 DataModule。
    
    ID: Normal, IR007, IR014, IR021, B007, B014, B021 (7 classes, 0HP)
    Shift: Same 7 classes at 3HP load
    OOD: OR007, OR014, OR021 (Outer Race faults - never seen in training)
    """
    
    num_classes = 7  # ID classes
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["cwru_outer_race_ood"]

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
        self._split_done = False

        self.train_transform = build_transforms("train", normalize=self.normalize_type)
        self.val_transform = build_transforms("val", normalize=self.normalize_type)
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
            id_df = build_df_from_files(
                root, ID_FILES, ID_CLASSES, ID_LABELS, self.max_id_windows_per_file
            )
            # Test cut uses a fixed shuffle seed, independent of the model seed.
            pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed
            )
        if self.eval_ood:
            classes = self._ood_class_list()
            self.ood_df = build_df_from_files(
                root, ood_files_for_classes(classes), classes, OOD_LABELS,
                self.max_ood_windows_per_file,
            )
        if self.eval_shift:
            self.shift_df = build_df_from_files(
                root, SHIFT_FILES, ID_CLASSES, ID_LABELS,
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
            dm = CWRUDataModule(root="/mnt/d/Data/Machine/CWRU", batch_size=64,
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
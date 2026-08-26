# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
from pathlib import Path
from typing import Literal
import warnings



import sys
# ──────────────────────────────────────────────────────────────
# 直接运行本文件时（不管从项目根目录还是 src/datasets 目录），
# 自动把项目源码目录加入 sys.path，保证本地模块可以直接导入，
# 不依赖外部 PYTHONPATH / .env 配置。
# 路径基于 __file__ 推算（本文件位于 src/datasets/thu.py），
# 与运行时的工作目录无关。
# ──────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_PROJECT_ROOT = _SRC_DIR.parent  # .../TFD，保证 `from src.datasets...` 也能解析
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.cache import load_cached_array

from src.datasets.base_dataset import dataset
from src.datasets.utils import (
    assert_disjoint_sources,
    random_pool_test_split,
    stratified_split,
)
from src.datasets.noise import NoisyEvaluationMixin
from src.datasets.transforms import build_transforms



ALL_CHANNELS = [
    "speed", "torque",
    "motor_vibration_x", "motor_vibration_y", "motor_vibration_z",
    "gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z",]
USE_CHANNELS = ["gearbox_vibration_y"]
TORQUE_SCALE = 6.0
SIGNAL_SIZE = 1024
SUBFOLDER = ""


# ID 类别（5 类）
ID_PREFIXES = [
    "health_",          # label 0
    "gear_pitting_H_",  # label 1
    "gear_wear_H_",     # label 2
    "teeth_break_H_",   # label 3
    "teeth_crack_H_",   # label 4
]
ID_LABELS = list(range(len(ID_PREFIXES)))  # 0~4

# OOD 类别（复合故障，仅用于评估，不参与训练）
OOD_PREFIXES = [
    "teeth_break_and_bearing_inner_H_",
    "teeth_break_and_bearing_outer_H_",
]
OOD_LABEL = -1
OOD_LABELS = [OOD_LABEL] * len(OOD_PREFIXES)

# OOD 按故障部位拆成两档，用于在集合过大时缩减评估样本数：
#   near —— 复合内圈轴承故障
#   far  —— 复合外圈轴承故障
# 两者是不同故障部位而非严重度分层，划分依据类别而非难度。
OOD_PREFIXES_NEAR = ["teeth_break_and_bearing_inner_H_"]
OOD_PREFIXES_FAR = ["teeth_break_and_bearing_outer_H_"]

# ──────────────────────────────────────────────────────────────
# 显式文件名分配（按文件级隔离，避免同一文件同时出现在多个 split）
# ──────────────────────────────────────────────────────────────
# train 用 speed_circulation（转速扫描），val/test 用 torque_circulation（扭矩扫描）
# 两类扫描方式文件互不重叠，天然保证 split 间零泄漏
TRAIN_SUFFIXES = [
    "speed_circulation_10Nm-1000rpm.csv",
    # "speed_circulation_20Nm-1000rpm.csv",
    # "speed_circulation_10Nm-2000rpm.csv",
    # "speed_circulation_20Nm-3000rpm.csv",
]
VAL_SUFFIXES = [
    "torque_circulation_2000rpm_20Nm.csv",
]
TEST_SUFFIXES = [
    "torque_circulation_3000rpm_10Nm.csv",
]
# OOD 工况与 test 对齐（同一工况、不同故障前缀），排除工况偏移这一混淆因素
OOD_SUFFIXES = [
    "torque_circulation_3000rpm_10Nm.csv",
]


# 与其余数据集统一命名：max_id_windows_per_file 统一控制 train/val/test 三个
# ID split 的单文件窗口上限（原来分别是 800/200/200，现改为单一可配置值，
# 默认取原 train 的量级）；max_ood_windows_per_file 控制 OOD split。
MAX_ID_WINDOWS_PER_FILE = 800
MAX_OOD_WINDOWS_PER_FILE = 200
TEST_RATIO = 0.2




def _read_multichannel_csv(
    path: Path,
    use_channels: list[str] = USE_CHANNELS,
    torque_scale: float = TORQUE_SCALE,
) -> np.ndarray:
    df = pd.read_csv(path, header=0, names=ALL_CHANNELS,
                      dtype=float, on_bad_lines="skip")
    if "torque" in use_channels and "torque" in df.columns:
        df["torque"] = df["torque"] * torque_scale
    return df[use_channels].dropna().to_numpy(dtype=np.float32)

def load_multichannel_csv(
    path: Path,
    use_channels: list[str] = USE_CHANNELS,
    torque_scale: float = TORQUE_SCALE,
) -> np.ndarray:
    """Load selected channels through the shared disk cache."""
    return load_cached_array(
        path,
        "thu",
        lambda source: _read_multichannel_csv(
            source, use_channels, torque_scale
        ),
        parameters={
            "format": "csv",
            "channels": list(use_channels),
            "torque_scale": torque_scale,
        },
    )



def slice_windows(
    arr: np.ndarray,
    label: int,
    win: int = SIGNAL_SIZE,
    max_windows: int | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """无重叠切窗。max_windows 限制该文件最多取多少个窗口（均匀间隔采样）。"""
    n = arr.shape[0]
    n_full_windows = n // win
    if n_full_windows == 0:
        return [], []

    all_starts = [i * win for i in range(n_full_windows)]

    if max_windows is not None and n_full_windows > max_windows:
        idx = np.linspace(0, n_full_windows - 1, max_windows, dtype=int)
        starts = [all_starts[i] for i in idx]
    else:
        starts = all_starts

    data = [arr[s: s + win] for s in starts]
    labels = [label] * len(data)
    return data, labels


def data_load(
    path: Path,
    label: int,
    use_channels: list[str] = USE_CHANNELS,
    torque_scale: float = TORQUE_SCALE,
    win: int = SIGNAL_SIZE,
    max_windows: int | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    if not path.exists():
        warnings.warn(f"[THUDataModule] 文件不存在，已跳过：{path}")
        return [], []
    arr = load_multichannel_csv(path, use_channels, torque_scale)
    return slice_windows(arr, label, win=win, max_windows=max_windows)


def build_df_from_files(
    root: str | Path,
    subfolder: str,
    prefixes: list[str],
    labels: list[int],
    suffixes: list[str],
    use_channels: list[str] = USE_CHANNELS,
    torque_scale: float = TORQUE_SCALE,
    win: int = SIGNAL_SIZE,
    max_windows_per_file: int | None = None,
) -> pd.DataFrame:
    folder = Path(root) / subfolder if subfolder else Path(root)
    all_data, all_labels, all_sources = [], [], []

    for prefix, label in zip(prefixes, labels):
        for suffix in suffixes:
            fname = f"{prefix}{suffix}"
            path = folder / fname
            d, l = data_load(
                path, label, use_channels, torque_scale, win,
                max_windows=max_windows_per_file,
            )
            if not d:
                warnings.warn(f"[THUDataModule] 未读取到数据：{fname}")
            all_data += d
            all_labels += l
            all_sources += [fname] * len(d)

    if not all_data:
        raise RuntimeError(
            f"[THUDataModule] 未加载到任何数据 (prefixes={prefixes}, suffixes={suffixes})")
    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })



class THUDataModule(NoisyEvaluationMixin, TUDataModule):
    """MCC5-THU 齿轮箱数据模块

    数据划分策略
    ─────────────
    split_mode="temporal"（默认）：采用文件级隔离而非窗口级随机切分，避免同一
    连续信号被拆入多个 split：
        train : speed_circulation 4 个工况文件（10/20Nm × 1000/2000rpm + 20Nm-3000rpm）
        val   : torque_circulation_2000rpm_20Nm.csv（1 个文件）
        test  : torque_circulation_3000rpm_10Nm.csv（1 个文件）
        ood   : 与 test 完全相同的工况后缀，但使用 OOD 故障前缀
                （工况对齐评估，排除工况偏移这一混淆因素）

    train 使用 speed_circulation、val/test 使用 torque_circulation，
    两类扫描方式来自不同文件，天然保证 split 之间零重叠。

    split_mode="random"：打散 train/val/test 的工况文件边界，将全部 ID 文件
    池化后按类别分层随机切分（与 seu.py/mgb.py 的 random 模式一致）。

    样本数控制
    ──────────
    max_id_windows_per_file：ID 相关 split（train/val/test）每个文件最多取多少
    窗口（均匀间隔采样，None = 不限）。
    max_ood_windows_per_file：OOD split 每个文件的窗口上限。
    """

    num_classes = len(ID_PREFIXES)
    num_channels = len(USE_CHANNELS)
    input_shape = (len(USE_CHANNELS), SIGNAL_SIZE)
    training_task = "classification"
    ood_datasets = ["thu_compound_ood"]

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
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # ── 数据集特有参数 ──────────────────────────
        subfolder: str = SUBFOLDER,
        use_channels: list[str] = USE_CHANNELS,
        torque_scale: float = TORQUE_SCALE,
        signal_size: int = SIGNAL_SIZE,
        id_prefixes: list[str] = ID_PREFIXES,
        ood_prefixes: list[str] = OOD_PREFIXES,
        train_suffixes: list[str] = TRAIN_SUFFIXES,
        val_suffixes: list[str] = VAL_SUFFIXES,
        test_suffixes: list[str] = TEST_SUFFIXES,
        ood_suffixes: list[str] = OOD_SUFFIXES,
        normalize_type: str = "-1-1",
        max_id_windows_per_file: int | None = MAX_ID_WINDOWS_PER_FILE,
        max_ood_windows_per_file: int | None = MAX_OOD_WINDOWS_PER_FILE,
        split_seed: int = 42,
        split_mode: Literal["random", "temporal"] = "temporal",
        ood_subset: Literal["all", "near", "far"] = "all",
        eval_noise: bool = False,
        noise_configs: list[tuple[str, int]] | None = None,
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
        self.subfolder = subfolder
        self.use_channels = use_channels
        self.torque_scale = torque_scale
        self.signal_size = signal_size
        self.id_prefixes = id_prefixes
        self.ood_prefixes = ood_prefixes
        self.train_suffixes = train_suffixes
        self.val_suffixes = val_suffixes
        self.test_suffixes = test_suffixes
        self.ood_suffixes = ood_suffixes
        self.normalize_type = normalize_type
        self.max_id_windows_per_file = max_id_windows_per_file
        self.max_ood_windows_per_file = max_ood_windows_per_file
        self.split_seed = split_seed
        self.split_mode = split_mode
        self.ood_subset = ood_subset
        self._split_done = False

        self.num_channels = len(use_channels)
        self.input_shape = (len(use_channels), signal_size)

        self.train_transform = build_transforms("train", normalize=self.normalize_type)
        self.val_transform = build_transforms("val", normalize=self.normalize_type)
        self.test_transform = build_transforms("val", normalize=self.normalize_type)
        self.ood_transform = build_transforms("val", normalize=self.normalize_type)

    def _build(self, prefixes: list[str], labels: list[int], suffixes: list[str], max_windows_per_file: int | None = None) -> pd.DataFrame:
        return build_df_from_files(
            self.root,
            self.subfolder,
            prefixes,
            labels,
            suffixes,
            self.use_channels,
            self.torque_scale,
            self.signal_size,
            max_windows_per_file=max_windows_per_file,
        )

    def _ood_prefix_list(self) -> list[str]:
        if self.ood_subset == "near":
            return OOD_PREFIXES_NEAR
        if self.ood_subset == "far":
            return OOD_PREFIXES_FAR
        return self.ood_prefixes

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if getattr(self, "_noisy_mode", False):
            return
        id_labels = list(range(len(self.id_prefixes)))
        if not self._split_done:
            if self.split_mode == "temporal":
                self.train_df = self._build(
                    self.id_prefixes, id_labels, self.train_suffixes,
                    max_windows_per_file=self.max_id_windows_per_file,
                )
                self.val_df = self._build(
                    self.id_prefixes, id_labels, self.val_suffixes,
                    max_windows_per_file=self.max_id_windows_per_file,
                )
                self.test_df = self._build(
                    self.id_prefixes, id_labels, self.test_suffixes,
                    max_windows_per_file=self.max_id_windows_per_file,
                )
                assert_disjoint_sources(self.train_df, self.val_df, self.test_df)
            else:
                if not self.val_split:
                    raise ValueError(
                        "val_split must be positive when split_mode='random'."
                    )
                # 打散 train/val/test 的工况文件边界，池化后按类别随机切分。
                all_suffixes = self.train_suffixes + self.val_suffixes + self.test_suffixes
                id_df = self._build(
                    self.id_prefixes, id_labels, all_suffixes,
                    max_windows_per_file=self.max_id_windows_per_file,
                )
                # Test cut uses a fixed shuffle seed, independent of the model seed.
                pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
                self.train_df, self.val_df = stratified_split(
                    pool_df, self.val_split, self.split_seed
                )
            if self.eval_ood:
                ood_prefixes = self._ood_prefix_list()
                self.ood_df = self._build(
                    ood_prefixes,
                    [OOD_LABEL] * len(ood_prefixes),
                    self.ood_suffixes,
                    max_windows_per_file=self.max_ood_windows_per_file,
                )
            self._split_done = True

        self.train = dataset(list_data=self.train_df, transform=self.train_transform)
        self.val = dataset(list_data=self.val_df, transform=self.val_transform)
        if stage in ("test", None):
            self.test = dataset(list_data=self.test_df, transform=self.test_transform)
        if self.eval_ood:
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

    def _data_loader(
        self,
        ds,
        training: bool = False,
        shuffle: bool = False,
    ) -> DataLoader:
        """覆盖父类方法，为 shuffle 操作绑定固定 seed 的 Generator，保证跨 run 可复现。"""
        generator = None
        if training or shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.split_seed)

        return DataLoader(
            ds,
            batch_size=self.batch_size if training else (self.eval_batch_size or self.batch_size),
            shuffle=training or shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            generator=generator,
        )

    def test_dataloader(self) -> list[DataLoader]:
        loaders = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            loaders.append(
                self._data_loader(self.get_ood_set(), training=False, shuffle=False)
            )
        return loaders




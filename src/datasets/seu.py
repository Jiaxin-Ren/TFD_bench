# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
import hashlib
import sys, warnings

warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from itertools import islice
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.cache import load_cached_array
from src.datasets.datamodule import TUDataModule
from src.datasets.base_dataset import dataset
from src.datasets.noise import DEFAULT_NOISE_PARAMS as NOISE_PARAMS, NoisyEvaluationMixin
from src.datasets.utils import (
    assert_disjoint_temporal_splits,
    random_pool_test_split,
    stratified_split,
)
from src.datasets.transforms import build_transforms
from typing import Literal

signal_size = 1024

ID_FILES = ["ball_20_0.csv", "comb_20_0.csv", "health_20_0.csv",
            "inner_20_0.csv", "outer_20_0.csv"]
SHIFT_FILES = ["ball_30_2.csv", "comb_30_2.csv", "health_30_2.csv",
               "inner_30_2.csv", "outer_30_2.csv"]

# OOD 拆成两档：
#   near —— 与 ID 同工况(20_0)，只差部件(轴承→齿轮)
#   far  —— 同时差部件与工况(30_2)，难度更低
# 混在一起会让 AUROC 成为双峰混合体的平均值，是之前跨 seed 方差巨大的原因之一。
OOD_FILES_NEAR = ["Chipped_20_0.csv", "Miss_20_0.csv",
                  "Root_20_0.csv", "Surface_20_0.csv"]
OOD_FILES_FAR = ["Chipped_30_2.csv", "Miss_30_2.csv",
                 "Root_30_2.csv", "Surface_30_2.csv"]
OOD_FILES = OOD_FILES_NEAR + OOD_FILES_FAR
_ALL_GEAR_FILES = set(OOD_FILES_NEAR) | set(OOD_FILES_FAR)

ID_LABELS = list(range(len(ID_FILES)))  # 0~4
SHIFT_LABELS = ID_LABELS                # shift 同为 5 类，标签与 ID 对齐

# OOD 样本不参与分类，标签统一置 -1，避免被误当作合法类别索引。
OOD_LABEL = -1

# OOD 集合降采样：每文件保留的最大窗口数。None 表示不限制。
# 250 × 4(near) = 1000，与 ID 测试集规模相当。
MAX_OOD_WINDOWS_PER_FILE: int | None = 250

# 数据划分专用种子。与 seed_everything 的全局 RNG 完全解耦，
# 保证不同 model seed 拿到完全相同的 train / val / test。
SPLIT_SEED = 12345
TEST_RATIO = 0.2

# temporal 切分时，各段之间丢弃的窗口数，切断时间上的相邻性。
SPLIT_GAP_WINDOWS = 1


# --------------------------------------------------------------------------- #
# 读取与切窗
# --------------------------------------------------------------------------- #

def load_signal_csv(filename: str, dataname: str) -> np.ndarray:
    fl = []
    # SEU 原始数据中仅 ball_20_0.csv 为逗号分隔，其余为制表符分隔。
    delim = "," if dataname == "ball_20_0.csv" else "\t"

    with open(filename, "r", errors="ignore") as f:
        for line in islice(f, 16, None):
            word = line.rstrip().split(delim)
            fl.append(float(word[1]))

    return np.array(fl).reshape(-1, 1)


def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]

    while end <= max_len:
        data.append(arr[start:end])
        labels.append(label)
        start += win
        end += win

    return data, labels


def subsample_uniform(items: list, n_max: int | None) -> list:
    """等间隔抽取至多 n_max 个元素，覆盖整段录制，完全确定性。"""
    if n_max is None or len(items) <= n_max:
        return items
    idx = np.linspace(0, len(items) - 1, n_max).round().astype(int)
    return [items[i] for i in idx]


def load_signal_cached(filename: str, dataname: str) -> np.ndarray:
    delim = "comma" if dataname == "ball_20_0.csv" else "tab"
    return load_cached_array(
        filename,
        "seu",
        lambda source: load_signal_csv(str(source), dataname),
        parameters={"skip_rows": 16, "column": 1, "delimiter": delim},
    )


def data_load(filename: str, dataname: str, label: int):
    arr = load_signal_cached(filename, dataname)
    return slice_windows(arr, label, win=signal_size)


def _subdir_for(fname: str) -> str:
    """齿轮故障文件位于 gearset/，轴承文件位于 bearingset/。"""
    return "gearset" if fname in _ALL_GEAR_FILES else "bearingset"


def build_df_from_files(root: str | Path, file_list, label_list,
                        max_per_file: int | None = None) -> pd.DataFrame:
    root = Path(root)
    all_data, all_labels, all_sources = [], [], []

    for fname, lbl in zip(file_list, label_list):
        path = root / _subdir_for(fname) / fname
        d, l = data_load(str(path), fname, lbl)
        if max_per_file is not None:
            d = subsample_uniform(d, max_per_file)
            l = l[:len(d)]
        all_data += d
        all_labels += l
        all_sources += [fname] * len(d)

    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


def build_split_by_time(root: str | Path, file_list, label_list,
                        test_ratio: float = TEST_RATIO,
                        gap_windows: int = SPLIT_GAP_WINDOWS):
    """按文件内时间顺序切分：每个文件前 (1-r) 段进 pool，后 r 段进 test。

    完全确定性，不使用随机数。切分点前丢弃 gap_windows 个窗作为缓冲，
    确保 pool 与 test 的窗口在时间上不相邻，避免同段录制导致的泄漏。
    """
    root = Path(root)
    pool_d, pool_l, pool_s, pool_i = [], [], [], []
    test_d, test_l, test_s, test_i = [], [], [], []

    for fname, lbl in zip(file_list, label_list):
        path = root / _subdir_for(fname) / fname
        d, l = data_load(str(path), fname, lbl)
        cut = int(len(d) * (1 - test_ratio))
        keep = max(cut - gap_windows, 0)
        pool_d += d[:keep]
        pool_l += l[:keep]
        test_d += d[cut:]
        test_l += l[cut:]
        pool_s += [fname] * keep
        pool_i += list(range(keep))
        test_s += [fname] * len(d[cut:])
        test_i += list(range(cut, len(d)))

    return (
        pd.DataFrame({
            "data": pool_d, "label": pool_l,
            "source": pool_s, "window_index": pool_i,
        }),
        pd.DataFrame({
            "data": test_d, "label": test_l,
            "source": test_s, "window_index": test_i,
        }),
    )


def temporal_val_split(df: pd.DataFrame, val_ratio: float,
                       gap_windows: int = SPLIT_GAP_WINDOWS):
    """按类别在时间顺序上切 train/val：每类后 val_ratio 段作为 val。

    要求 df 内同类样本仍保持时间顺序（build_split_by_time 的输出满足）。
    切分点前同样丢弃 gap_windows 个窗，避免 train 与 val 时间相邻。
    """
    labels = df["label"].to_numpy()
    train_pos, val_pos = [], []
    for lbl in np.unique(labels):
        pos = np.flatnonzero(labels == lbl)      # 已按时间升序
        cut = int(round(len(pos) * (1 - val_ratio)))
        keep = max(cut - gap_windows, 0)
        train_pos.append(pos[:keep])
        val_pos.append(pos[cut:])
    train_pos = np.sort(np.concatenate(train_pos))
    val_pos = np.sort(np.concatenate(val_pos))
    return (df.iloc[train_pos].reset_index(drop=True),
            df.iloc[val_pos].reset_index(drop=True))


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #

class SEUDataModule(NoisyEvaluationMixin, TUDataModule):
    num_classes = 5
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["seu_gear_ood"]
    noise_params = NOISE_PARAMS

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
            split_seed: int = SPLIT_SEED,
            split_mode: Literal["random", "temporal"] = "temporal",
            ood_subset: Literal["all", "near", "far"] = "near",
            max_ood_windows_per_file: int | None = MAX_OOD_WINDOWS_PER_FILE,
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
        self.noise_configs = noise_configs or [(t, s) for t in NOISE_PARAMS for s in range(1, 6)]
        self.normalize_type = normalize_type
        self.split_seed = split_seed
        self.split_mode = split_mode
        self.ood_subset = ood_subset
        self.max_ood_windows_per_file = max_ood_windows_per_file
        self.max_shift_windows_per_file = max_shift_windows_per_file
        self._split_done = False

        self.train_transform = build_transforms("train", normalize=self.normalize_type)
        self.val_transform = build_transforms("val", normalize=self.normalize_type)
        self.test_transform = build_transforms("val", normalize=self.normalize_type)
        self.ood_transform = build_transforms("val", normalize=self.normalize_type)

    # ----------------------------------------------------------------- #

    def _ood_file_list(self) -> list[str]:
        if self.ood_subset == "near":
            return OOD_FILES_NEAR
        if self.ood_subset == "far":
            return OOD_FILES_FAR
        return OOD_FILES

    def _build_splits(self) -> None:
        """只执行一次。fit 与 test 阶段共用同一份划分。"""
        if self._split_done:
            return

        root = Path(self.root)

        if not self.val_split:
            raise ValueError(
                "val_split 必须为正数。验证集不能等于测试集，"
                "否则模型选择与后处理校准都会泄漏测试数据。"
            )

        if self.split_mode == "temporal":
            # 时间切分：pool / test 按录制顺序分开，中间留缓冲窗。
            # train / val 同样按时间切，保证三者互不相邻。
            pool_df, self.test_df = build_split_by_time(
                root, ID_FILES, ID_LABELS, TEST_RATIO)
            self.train_df, self.val_df = temporal_val_split(pool_df, self.val_split)
            assert_disjoint_temporal_splits(
                self.train_df, self.val_df, self.test_df)
        else:
            id_df = build_df_from_files(root, ID_FILES, ID_LABELS)
            # 测试集划分固定为 random_state=42，与 model seed 无关
            pool_df, self.test_df = random_pool_test_split(id_df, TEST_RATIO)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed)

        if self.eval_ood:
            files = self._ood_file_list()
            self.ood_df = build_df_from_files(
                root, files, [OOD_LABEL] * len(files),
                max_per_file=self.max_ood_windows_per_file)
        if self.eval_shift:
            self.shift_df = build_df_from_files(
                root, SHIFT_FILES, SHIFT_LABELS,
                max_per_file=self.max_shift_windows_per_file)

        self._split_done = True

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        # 噪声评估模式下跳过 setup，防止覆盖已替换的 test / ood
        if getattr(self, '_noisy_mode', False):
            return

        self._build_splits()

        self.train = dataset(list_data=self.train_df, transform=self.train_transform)
        self.val = dataset(list_data=self.val_df, transform=self.test_transform)

        if stage in ("test", None):
            self.test = dataset(list_data=self.test_df, transform=self.test_transform)

        if self.eval_ood:
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

        if self.eval_shift:
            self.shift = dataset(list_data=self.shift_df, transform=self.test_transform)

    # ----------------------------------------------------------------- #

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
        dataloaders = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            dataloaders.append(
                self._data_loader(self.get_ood_set(), training=False, shuffle=False))
        if self.eval_shift:
            dataloaders.append(
                self._data_loader(self.get_shift_set(), training=False, shuffle=False))
        return dataloaders



if __name__ == "__main__":
    import torch

    for mode in ["random", "temporal"]:
        print(f"\n{'=' * 60}\n  split_mode = {mode}\n{'=' * 60}")
        fingerprints = []
        for s in [0, 1, 2]:
            torch.manual_seed(s)
            np.random.seed(s)
            dm = SEUDataModule(root="/mnt/d/Data/Machine/SEU", batch_size=64,
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
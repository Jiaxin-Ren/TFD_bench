# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
Shared utilities for benchmark scripts.
Benchmark 脚本的共享工具模块。

Includes GPU configuration parsing and common argument setup.
包含 GPU 配置解析和通用参数设置。

Configuration Priority / 配置优先级:
1. Command line arguments (highest) / 命令行参数（最高）
2. YAML config file / YAML配置文件
3. Default values (lowest) / 默认值（最低）
"""

import warnings
import logging
import os

# Suppress all Python warnings
warnings.filterwarnings("ignore")

# Suppress verbose logging from Lightning, transformers, etc.
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress Lightning/PyTorch environment warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import torch

# Default config file path / 默认配置文件路径
# Calculate absolute path relative to this file's location
_UTILS_DIR = Path(__file__).parent  # src/
_PROJECT_ROOT = _UTILS_DIR.parent   # benchmark_tfd/
DEFAULT_CONFIG_PATH = str(_PROJECT_ROOT / "configs" / "default.yaml")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    从YAML文件加载配置。
    
    Args:
        config_path: Path to YAML config file. If None, uses default path.
        
    Returns:
        Dict with configuration values
    """
    if config_path is None:
        # Try default config path
        config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


def add_common_args(parser: argparse.ArgumentParser, config_path: str = None) -> argparse.ArgumentParser:
    """
    Add common arguments to parser with defaults from config file.
    添加通用参数到解析器，默认值从配置文件读取。
    
    Config file is loaded first, then command line can override.
    先加载配置文件，命令行参数可覆盖。
    """
    # Resolve --config before constructing the main parser so that every
    # default comes from the user-selected file rather than always default.yaml.
    if config_path is None:
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
        config_path, _ = config_parser.parse_known_args()
        config_path = config_path.config

    # Load config for defaults
    config = load_config(config_path)
    configured_datasets = config.get("datasets") or []
    dataset_cfg = (
        configured_datasets[0]
        if configured_datasets and isinstance(configured_datasets[0], dict)
        else config.get("dataset", {})
    )
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    gpu_cfg = config.get("hardware", config.get("gpu", {}))
    
    # Backbone
    parser.add_argument("--backbone", type=str, 
                        default=config.get("backbones", ["resnet"])[0] if config.get("backbones") else "resnet",
                        choices=["resnet", "lenet", "mlp", "transformer",
                                 "lstm", "timesnet"],
                        help="Backbone architecture / 骨干网络架构")
    
    # Dataset - defaults from config
    dataset_name = dataset_cfg.get("name", "seu")
    dataset_roots = config.get("dataset_roots", {})
    default_data_root = dataset_cfg.get("root", dataset_roots.get(dataset_name, "."))
    
    parser.add_argument("--dataset", type=str, 
                        default=dataset_name,
                        choices=["seu", "wt", "pu", "xjtu", "hit", "cwru", "thu", "mgb"],
                        help="Dataset to use / 使用的数据集")
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str,
                        default=default_data_root,
                        help="Path to dataset / 数据集路径")
    # Training - defaults from config
    parser.add_argument("--epochs", type=int, 
                        default=training_cfg.get("epochs", 30), 
                        help="Number of epochs / 训练轮数")
    parser.add_argument(
        "--max-id-windows-per-file",
        type=int,
        default=dataset_cfg.get("max_id_windows_per_file"),
        help="Deterministic per-file cap applied before ID splitting.",
    )
    parser.add_argument(
        "--max-ood-windows-per-file",
        type=int,
        default=dataset_cfg.get("max_ood_windows_per_file"),
        help="Deterministic per-file cap for OOD evaluation.",
    )
    parser.add_argument(
        "--max-shift-windows-per-file",
        type=int,
        default=dataset_cfg.get("max_shift_windows_per_file"),
        help="Deterministic per-file cap for operating-shift evaluation.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=dataset_cfg.get("split_mode"),
        choices=["random", "temporal"],
        help="ID train/val/test split strategy (defaults to the dataset's own default).",
    )
    parser.add_argument(
        "--ood-subset",
        type=str,
        default=dataset_cfg.get("ood_subset"),
        choices=["all", "near", "far"],
        help="Which OOD fault subset to evaluate (defaults to the dataset's own default).",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int,
                        default=training_cfg.get("batch_size", 48), 
                        help="Batch size / 批次大小")
    parser.add_argument("--lr", type=float, 
                        default=training_cfg.get("learning_rate", 1e-3), 
                        help="Learning rate / 学习率")
    
    # GPU Configuration - defaults from config
    parser.add_argument("--devices", type=str, 
                        default=str(gpu_cfg.get("devices", 1)),
                        help="Number of GPUs: 1, 2, 'auto' for all / GPU数量: 1, 2, 'auto'表示全部")
    parser.add_argument("--strategy", type=str, 
                        default=gpu_cfg.get("strategy", "auto"),
                        choices=["auto", "ddp", "dp", "ddp_spawn", "fsdp"],
                        help="Distributed strategy / 分布式策略")
    parser.add_argument("--accelerator", type=str, 
                        default=gpu_cfg.get("accelerator", "auto"),
                        choices=["auto", "gpu", "cpu"],
                        help="Accelerator type / 加速器类型")
    parser.add_argument("--precision", type=str, 
                        default=str(gpu_cfg.get("precision", "32")),
                        choices=["32", "16", "bf16-mixed", "16-mixed"],
                        help="Training precision / 训练精度")
    
    # Config file path (for explicit override)
    parser.add_argument("--config", type=str, default=config_path or DEFAULT_CONFIG_PATH,
                        help="Path to YAML config file / YAML配置文件路径")
    
    parser.add_argument("--eval-noise", "--eval_noise", dest="eval_noise", action="store_true",
                        default=evaluation_cfg.get("noise", dataset_cfg.get("eval_noise", False)),
                        help="评估噪声鲁棒性")
    parser.add_argument("--no-eval-noise", dest="eval_noise", action="store_false",
                        help="禁用噪声鲁棒性评估")
    parser.add_argument("--eval-shift", "--eval_shift", dest="eval_shift", action="store_true",
                        default=evaluation_cfg.get("shift", dataset_cfg.get("eval_shift", False)),
                        help="评估运行工况引起的分布偏移")
    parser.add_argument("--no-eval-shift", dest="eval_shift", action="store_false",
                        help="禁用运行工况分布偏移评估")
    parser.add_argument("--output-dir", type=str,
                        default=config.get("output", {}).get("dir", "results"),
                        help="Experiment output directory / 实验输出目录")
    
    return parser


def get_datamodule(args: argparse.Namespace, **kwargs):
    """
    Get the appropriate DataModule based on args.
    根据参数获取合适的 DataModule。
    
    Usage:
        datamodule = get_datamodule(args, eval_ood=True, eval_shift=True)
    
    Args:
        args: Parsed arguments with dataset, data_root, and batch_size
        **kwargs: Additional arguments passed to DataModule (e.g., eval_ood, eval_shift)
    
    Returns:
        DataModule instance
    """
    dataset_name = getattr(args, 'dataset', 'seu').lower()
    # All datasets accept max_id/ood_windows_per_file, split_mode and ood_subset.
    # thu is the only dataset without shift-evaluation support, so it alone
    # does not accept max_shift_windows_per_file.
    forwarded_options = [
        "max_id_windows_per_file",
        "max_ood_windows_per_file",
        "split_mode",
        "ood_subset",
    ]
    if dataset_name != "thu":
        forwarded_options.append("max_shift_windows_per_file")
    for option in forwarded_options:
        value = getattr(args, option, None)
        if value is not None:
            kwargs.setdefault(option, value)
    # Get data root
    root = Path(args.data_root)
    
    # Import the selected 1D time-domain dataset.
    if dataset_name == "seu":
        from src.datasets.seu import SEUDataModule
        DataModuleClass = SEUDataModule
    
    elif dataset_name == "wt":
        from src.datasets.wt import WTDataModule
        DataModuleClass = WTDataModule
    
    elif dataset_name == "pu":
        from src.datasets.pu import PUDataModule
        DataModuleClass = PUDataModule
    
    elif dataset_name == "xjtu":
        from src.datasets.xjtu import XJTUDataModule
        DataModuleClass = XJTUDataModule
    
    elif dataset_name == "cwru":
        from src.datasets.cwru import CWRUDataModule
        DataModuleClass = CWRUDataModule

    elif dataset_name == "thu":
        from src.datasets.thu import THUDataModule
        DataModuleClass = THUDataModule

    elif dataset_name == "mgb":
        from src.datasets.mgb import MGBDataModule
        DataModuleClass = MGBDataModule
    
    elif dataset_name == "hit":
        # HIT supports channel selection via args.channels (e.g., [0] or [0,1,2])
        channels = getattr(args, 'channels', None)
        from src.datasets.hit import HITDataModule
        return HITDataModule(
            root=root,
            batch_size=args.batch_size,
            channels=channels,
            **kwargs
        )
        
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: "
            "seu, wt, pu, xjtu, hit, cwru, thu, mgb"
        )
    
    # Create and return datamodule
    return DataModuleClass(
        root=root,
        batch_size=args.batch_size,
        **kwargs
    )


def parse_devices(devices_str: str) -> int | str:
    """
    Parse devices string to appropriate type.
    解析 devices 字符串为适当类型。
    """
    if devices_str.lower() == "auto":
        return "auto"
    elif devices_str == "-1":
        return "auto"
    else:
        try:
            return int(devices_str)
        except ValueError:
            return devices_str


def load_gpu_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load GPU configuration from args and optional config file.
    从参数和可选配置文件加载 GPU 配置。
    
    Returns:
        Dict with keys: accelerator, devices, strategy, precision
    """
    # Default from args
    gpu_config = {
        "accelerator": args.accelerator if hasattr(args, "accelerator") else "auto",
        "devices": parse_devices(args.devices) if hasattr(args, "devices") else 1,
        "strategy": args.strategy if hasattr(args, "strategy") else "auto",
        "precision": args.precision if hasattr(args, "precision") else "32",
    }
    
    # Convert precision to int if possible
    if gpu_config["precision"] in ["32", "16"]:
        gpu_config["precision"] = int(gpu_config["precision"])
    
    return gpu_config


def get_trainer_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get keyword arguments for TUTrainer based on args.
    根据参数获取 TUTrainer 的关键字参数。
    
    Usage:
        trainer_kwargs = get_trainer_kwargs(args)
        trainer = TUTrainer(**trainer_kwargs, max_epochs=args.epochs, callbacks=[...])
    """
    gpu_config = load_gpu_config(args)
    
    trainer_kwargs = {
        "accelerator": gpu_config["accelerator"],
        "devices": gpu_config["devices"],
        "precision": gpu_config["precision"],
    }
    
    # Only add strategy if using multiple GPUs or explicit strategy
    if gpu_config["strategy"] != "auto":
        trainer_kwargs["strategy"] = gpu_config["strategy"]
    elif isinstance(gpu_config["devices"], int) and gpu_config["devices"] > 1:
        trainer_kwargs["strategy"] = "ddp"  # Default to DDP for multi-GPU
    elif gpu_config["devices"] == "auto":
        # Check available GPUs
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            trainer_kwargs["strategy"] = "ddp"
    
    return trainer_kwargs


def print_gpu_config(gpu_config: Dict[str, Any]):
    """Print GPU configuration / 打印 GPU 配置"""
    print("=" * 50)
    print("GPU Configuration / GPU 配置")
    print("=" * 50)
    print(f"  Accelerator: {gpu_config['accelerator']}")
    print(f"  Devices: {gpu_config['devices']}")
    print(f"  Strategy: {gpu_config.get('strategy', 'auto')}")
    print(f"  Precision: {gpu_config['precision']}")
    if torch.cuda.is_available():
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 50)


def print_dataset_config(args: argparse.Namespace):
    """Print dataset configuration / 打印数据集配置"""
    dataset = getattr(args, 'dataset', 'seu')
    print(f"  Dataset: {dataset.upper()}")
    print(f"  Data root: {args.data_root}")
    print("  Input type: 1D time domain")


def get_model(backbone_name: str, num_channels: int, num_classes: int, **kwargs) -> torch.nn.Module:
    """
    Get model based on backbone name.
    根据骨干网络名称获取模型。
    
    Supports 1D vibration-signal models.
    支持一维振动信号模型。
    """
    # 1D Models
    if backbone_name == "resnet":
        from src.models.resnet import resnet1d
        return resnet1d(in_channels=num_channels, num_classes=num_classes, **kwargs)
    elif backbone_name == "lenet":
        from src.models.lenet import lenet1d
        return lenet1d(in_channels=num_channels, num_classes=num_classes, **kwargs)
    elif backbone_name == "mlp":
        from src.models.mlp import mlp
        # Calculate in_features from input_shape if available
        input_shape = kwargs.get("input_shape", None)
        if input_shape:
            import numpy as np
            in_features = int(np.prod(input_shape))
        else:
            # Fallback: assume standard 1024 length if not provided (risky but allows basic usage)
            # Default signal length in this benchmark is often 1024 or 20480. 
            # Better to error if unknown, but for now defaulting to 1024 * C
            in_features = 1024 * num_channels
            
        # Default hidden dims if not provided
        hidden_dims = kwargs.get("hidden_dims", [128, 128])
        
        # Remove kwargs not accepted by mlp if necessary, or just pass them and let it error/ignore?
        # mlp accepts kwargs in common (activatiom, dropout_rate), but 'input_shape' needs to be removed from kwargs if we passed it
        kwargs_mlp = kwargs.copy()
        kwargs_mlp.pop("input_shape", None)
        kwargs_mlp.pop("hidden_dims", None)
        
        return mlp(in_features=in_features, num_outputs=num_classes, hidden_dims=hidden_dims, flatten_start_dim=1, **kwargs_mlp)
    elif backbone_name == "transformer":
        from src.models.transformer import transformer1d
        return transformer1d(in_channels=num_channels, num_classes=num_classes, **kwargs)
    elif backbone_name == "lstm":
        from src.models.lstm import LSTM1D 
        return LSTM1D(in_channels=num_channels, num_classes=num_classes, **kwargs)

    elif backbone_name == "timesnet":
        from src.models.timesnet import TimesNetWrapper
        from types import SimpleNamespace
        configs = SimpleNamespace(
            task_name='classification', seq_len=1024, enc_in=num_channels, num_class=num_classes,
            pred_len=0, label_len=0, d_model=32, d_ff=64, e_layers=1,  # 减小模型
            embed='timeF', freq='h', dropout=0.1, top_k=2, num_kernels=3, c_out=num_channels)
        return TimesNetWrapper(configs)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. Available: "
            "resnet, lenet, mlp, transformer, lstm, timesnet"
        )

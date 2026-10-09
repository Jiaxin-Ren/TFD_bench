"""Run the configured method × backbone benchmark matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from queue import Empty, Queue
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
STATUS_COLUMNS = (
    "index",
    "dataset",
    "method",
    "backbone",
    "gpu",
    "status",
    "returncode",
    "elapsed_seconds",
    "started_at",
    "finished_at",
    "log",
)

# These options belong to the shared experiment protocol.  Dataset-specific
# method overrides must not change them, otherwise methods would no longer be
# compared under the same training and evaluation conditions.
COMMON_ARGUMENTS = frozenset({
    "accelerator",
    "backbone",
    "batch_size",
    "channels",
    "ckpt",
    "config",
    "continue_on_error",
    "data_root",
    "dataset",
    "dataset_roots",
    "datasets",
    "devices",
    "epochs",
    "eval_noise",
    "eval_shift",
    "evaluation",
    "hardware",
    "gpus",
    "learning_rate",
    "lr",
    "max_id_windows_per_file",
    "max_ood_windows_per_file",
    "max_shift_windows_per_file",
    "n_runs",
    "name",
    "noise",
    "ood_subset",
    "output",
    "output_dir",
    "overwrite",
    "precision",
    "root",
    "resume",
    "runner",
    "seeds",
    "shift",
    "split_mode",
    "strategy",
    "training",
    "val_split",
    "workers",
})


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError("The configuration root must be a YAML mapping.")
    return config


def append_option(command: list[str], key: str, value: Any) -> None:
    """Translate a YAML method argument to its argparse representation."""
    option = "--" + key.replace("_", "-")
    if value is None:
        return
    if isinstance(value, bool):
        command.append(option if value else "--no-" + key.replace("_", "-"))
        return
    command.append(option)
    if isinstance(value, list):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def validate_method_dataset_args(value: Any) -> Mapping[str, Any]:
    """Validate per-dataset overrides without constraining method parameters."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("method_dataset_args must be a mapping.")

    for method, dataset_args in value.items():
        method_path = f"method_dataset_args.{method}"
        if not isinstance(dataset_args, Mapping):
            raise TypeError(f"{method_path} must be a mapping.")
        for dataset, overrides in dataset_args.items():
            dataset_path = f"{method_path}.{dataset}"
            if not isinstance(overrides, Mapping):
                raise TypeError(f"{dataset_path} must be a mapping.")
            non_string_keys = [key for key in overrides if not isinstance(key, str)]
            if non_string_keys:
                raise TypeError(f"{dataset_path} parameter names must be strings.")
            common = sorted(
                key for key in overrides
                if key.replace("-", "_") in COMMON_ARGUMENTS
            )
            if common:
                names = ", ".join(common)
                raise ValueError(
                    f"{dataset_path} may only contain method-specific arguments; "
                    f"common argument(s) are not allowed: {names}"
                )
    return value


def build_commands(config_path: Path, config: dict[str, Any]) -> list[list[str]]:
    methods = config.get("methods") or []
    backbones = config.get("backbones") or []
    datasets = config.get("datasets") or []
    if not datasets and config.get("dataset"):
        datasets = [config["dataset"]]
    method_args = config.get("method_args") or {}
    method_dataset_args = validate_method_dataset_args(
        config.get("method_dataset_args")
    )
    if not methods:
        raise ValueError("No methods selected in the configuration.")
    if not backbones:
        raise ValueError("No backbones selected in the configuration.")
    if not datasets:
        raise ValueError("No datasets selected in the configuration.")

    for dataset in datasets:
        if not isinstance(dataset, dict) or not dataset.get("name") or not dataset.get("root"):
            raise ValueError("Each dataset must define both name and root.")

    commands = []
    for method in methods:
        method_file = PROJECT_ROOT / "methods" / f"{method}.py"
        if not method_file.is_file():
            raise ValueError(f"Unknown method {method!r}: {method_file} does not exist.")
        method_overrides = method_args.get(method, {}) or {}
        if not isinstance(method_overrides, dict):
            raise TypeError(f"method_args.{method} must be a mapping.")
        dataset_overrides = method_dataset_args.get(method, {})
        for dataset in datasets:
            overrides = {
                **method_overrides,
                **dataset_overrides.get(str(dataset["name"]), {}),
            }
            for backbone in backbones:
                command = [
                    sys.executable,
                    str(method_file),
                    "--config", str(config_path),
                    "--dataset", str(dataset["name"]),
                    "--data-root", str(dataset["root"]),
                    "--backbone", str(backbone),
                ]
                for key, value in dataset.items():
                    if key not in {"name", "root"}:
                        append_option(command, key, value)
                for key, value in overrides.items():
                    append_option(command, key, value)
                commands.append(command)
    return commands


def command_identity(command: list[str]) -> tuple[str, str, str]:
    """Return ``(dataset, method, backbone)`` for a generated command."""
    method = Path(command[1]).stem
    dataset = command[command.index("--dataset") + 1]
    backbone = command[command.index("--backbone") + 1]
    return dataset, method, backbone


def parse_gpu_ids(value: str | list[int] | None, workers: int) -> list[int]:
    """Parse and validate the physical GPU ids assigned to runner workers."""
    if isinstance(value, list):
        gpu_ids = [int(item) for item in value]
    elif value:
        gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        gpu_ids = list(range(workers))
    if not gpu_ids:
        raise ValueError("At least one GPU id is required for parallel execution.")
    if len(set(gpu_ids)) != len(gpu_ids) or any(item < 0 for item in gpu_ids):
        raise ValueError(f"GPU ids must be unique non-negative integers: {gpu_ids}")
    return gpu_ids


def force_single_visible_gpu(command: list[str]) -> list[str]:
    """Keep each parallel worker on its one assigned visible GPU."""
    return [
        *command,
        "--accelerator", "gpu",
        "--devices", "1",
        "--strategy", "auto",
    ]


def result_dir(config: dict[str, Any], command: list[str]) -> Path:
    dataset, method, backbone = command_identity(command)
    output = Path((config.get("output") or {}).get("dir", "results"))
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    return output / dataset / backbone / method


def has_complete_result(config: dict[str, Any], command: list[str]) -> bool:
    """Return whether a complete result with its required tables is present."""
    output = result_dir(config, command)
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    dataset, method, backbone = command_identity(command)
    return (
        manifest.get("status") == "complete"
        and manifest.get("dataset") == dataset
        and manifest.get("method") == method
        and manifest.get("backbone") == backbone
        and (output / "runs.csv").is_file()
        and (output / "summary.csv").is_file()
    )


def _atomic_write_status(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: int(row["index"])))
    os.replace(temporary, path)


def run_parallel(
    commands: list[list[str]],
    config: dict[str, Any],
    *,
    gpu_ids: list[int],
    continue_on_error: bool,
    resume: bool,
) -> list[tuple[str, str, str, int]]:
    """Run a dynamic queue with one persistent single-GPU worker per GPU."""
    output_root = result_dir(config, commands[0]).parents[2]
    logs_dir = output_root / "_runner_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "runner_status.csv"

    queue: Queue[tuple[int, list[str]]] = Queue()
    for index, command in enumerate(commands, start=1):
        queue.put((index, command))

    total = len(commands)
    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str, int]] = []
    lock = threading.Lock()
    stop = threading.Event()

    def record(row: dict[str, Any]) -> None:
        with lock:
            records.append(row)
            _atomic_write_status(status_path, records)

    def worker(gpu_id: int) -> None:
        while not stop.is_set():
            try:
                index, command = queue.get_nowait()
            except Empty:
                return
            dataset, method, backbone = command_identity(command)
            log_path = logs_dir / f"{dataset}__{backbone}__{method}.log"
            started = datetime.now(timezone.utc)
            if resume and has_complete_result(config, command):
                record({
                    "index": index,
                    "dataset": dataset,
                    "method": method,
                    "backbone": backbone,
                    "gpu": gpu_id,
                    "status": "skipped_complete",
                    "returncode": 0,
                    "elapsed_seconds": "0.000",
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "log": str(log_path),
                })
                print(
                    f"[{index}/{total}] GPU {gpu_id} SKIP "
                    f"{dataset}/{method}/{backbone}",
                    flush=True,
                )
                queue.task_done()
                continue

            child_command = force_single_visible_gpu(command)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(
                f"[{index}/{total}] GPU {gpu_id} START "
                f"{dataset}/{method}/{backbone}",
                flush=True,
            )
            start_time = time.monotonic()
            returncode = -1
            error = None
            with log_path.open("w", encoding="utf-8") as stream:
                stream.write("COMMAND: " + " ".join(child_command) + "\n")
                stream.flush()
                try:
                    result = subprocess.run(
                        child_command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    returncode = result.returncode
                except OSError as exc:
                    error = exc
                    stream.write(f"RUNNER ERROR: {exc}\n")
            elapsed = time.monotonic() - start_time
            status = "complete" if returncode == 0 else "failed"
            finished = datetime.now(timezone.utc)
            record({
                "index": index,
                "dataset": dataset,
                "method": method,
                "backbone": backbone,
                "gpu": gpu_id,
                "status": status,
                "returncode": returncode,
                "elapsed_seconds": f"{elapsed:.3f}",
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "log": str(log_path),
            })
            print(
                f"[{index}/{total}] GPU {gpu_id} {status.upper()} "
                f"{dataset}/{method}/{backbone} ({elapsed:.1f}s)",
                flush=True,
            )
            if returncode != 0:
                with lock:
                    failures.append((dataset, method, backbone, returncode))
                if error is not None:
                    print(f"Runner could not start the command: {error}", flush=True)
                if not continue_on_error:
                    stop.set()
            queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"gpu-{gpu_id}")
        for gpu_id in gpu_ids[:min(len(gpu_ids), total)]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"Runner status: {status_path}", flush=True)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel single-GPU workers; defaults to runner.workers or 1.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids assigned to workers, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip combinations whose result manifest and tables are complete.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    commands = build_commands(config_path, config)
    runner_config = config.get("runner", {})
    continue_on_error = bool(runner_config.get("continue_on_error", True))
    workers = (
        args.workers
        if args.workers is not None
        else int(runner_config.get("workers", 1))
    )
    resume = (
        args.resume
        if args.resume is not None
        else bool(runner_config.get("resume", False))
    )
    configured_gpus = (
        args.gpus if args.gpus is not None else runner_config.get("gpus")
    )
    if workers < 1:
        parser.error("--workers must be at least 1")

    if workers > 1:
        gpu_ids = parse_gpu_ids(configured_gpus, workers)[:workers]
        if len(gpu_ids) < workers:
            parser.error(
                f"--workers={workers} requires at least {workers} GPU ids; "
                f"got {gpu_ids}"
            )
        if args.dry_run:
            for index, command in enumerate(commands, start=1):
                gpu = gpu_ids[(index - 1) % len(gpu_ids)]
                print(
                    f"[{index}/{len(commands)}] GPU {gpu} "
                    + " ".join(force_single_visible_gpu(command))
                )
            return 0
        failures = run_parallel(
            commands,
            config,
            gpu_ids=gpu_ids,
            continue_on_error=continue_on_error,
            resume=resume,
        )
        if failures:
            print("\nFailed experiments:")
            for dataset, method, backbone, returncode in failures:
                print(f"  {dataset} + {method} + {backbone}: exit code {returncode}")
            return 1
        return 0

    failures = []
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        dataset, method, backbone = command_identity(command)
        print(
            f"\n[{index}/{total}] dataset={dataset} method={method} backbone={backbone}",
            flush=True,
        )
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        if resume and has_complete_result(config, command):
            print("Skipping complete result.", flush=True)
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            failures.append((dataset, method, backbone, result.returncode))
            if not continue_on_error:
                break

    if failures:
        print("\nFailed experiments:")
        for dataset, method, backbone, returncode in failures:
            print(f"  {dataset} + {method} + {backbone}: exit code {returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

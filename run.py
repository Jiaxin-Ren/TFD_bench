"""Run the configured method × backbone benchmark matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


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


def build_commands(config_path: Path, config: dict[str, Any]) -> list[list[str]]:
    methods = config.get("methods") or []
    backbones = config.get("backbones") or []
    datasets = config.get("datasets") or []
    if not datasets and config.get("dataset"):
        datasets = [config["dataset"]]
    method_args = config.get("method_args") or {}
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
        overrides = method_args.get(method, {}) or {}
        if not isinstance(overrides, dict):
            raise TypeError(f"method_args.{method} must be a mapping.")
        for dataset in datasets:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    commands = build_commands(config_path, config)
    runner_config = config.get("runner", {})
    continue_on_error = bool(runner_config.get("continue_on_error", True))

    failures = []
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        method = Path(command[1]).stem
        dataset = command[command.index("--dataset") + 1]
        backbone = command[command.index("--backbone") + 1]
        print(
            f"\n[{index}/{total}] dataset={dataset} method={method} backbone={backbone}",
            flush=True,
        )
        print(" ".join(command), flush=True)
        if args.dry_run:
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

# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""Shared multi-run experiment infrastructure for all method entry points."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import roc_auc_score

from src.utils import get_datamodule, get_model, get_trainer_kwargs, load_config

from .classification import ClassificationRoutine
from .utils import TUTrainer
Metrics = dict[str, Any]
Results = dict[str, Metrics]
RESULT_SCHEMA_VERSION = 2


def require_ensemble_samples(
    model: nn.Module,
    method: str,
    expected: int | None = None,
) -> int:
    """Fail loudly when a sampling method reaches evaluation as a single model."""
    if hasattr(model, "saved_models"):
        count = len(model.saved_models) + int(bool(getattr(model, "use_final_model", False)))
    elif hasattr(model, "samples"):
        count = len(model.samples)
    else:
        value = getattr(model, "num_estimators", 0)
        count = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
    if count < 2:
        raise RuntimeError(
            f"{method} produced only {count} estimator(s); refusing to report "
            "ensemble uncertainty or OOD AUROC. Check the burn-in/collection schedule."
        )
    if expected is not None and count != expected:
        raise RuntimeError(
            f"{method} produced {count} estimators, but {expected} were requested. "
            "Increase the training epochs or adjust the collection schedule."
        )
    print(f"{method}: evaluating {count} posterior estimators")
    return count


def add_ood_source_metrics(
    metrics: Metrics,
    routine: ClassificationRoutine,
    datamodule: Any,
) -> Metrics:
    """Add rank-based overall, per-source, and macro OOD AUROC metrics."""
    if not getattr(datamodule, "eval_ood", False):
        return metrics

    artifacts = routine.get_prediction_artifacts()
    id_scores = artifacts.get("id_ood_scores")
    ood_scores = artifacts.get("ood_scores")
    ood_frame = getattr(datamodule, "ood_df", None)
    if id_scores is None or ood_scores is None or ood_frame is None:
        return metrics
    if "source" not in ood_frame or len(ood_frame) != ood_scores.numel():
        return metrics

    id_values = id_scores.detach().cpu().numpy().reshape(-1)
    ood_values = ood_scores.detach().cpu().numpy().reshape(-1)
    id_values = id_values[np.isfinite(id_values)]
    sources = ood_frame["source"].astype(str).to_numpy()
    finite_ood = np.isfinite(ood_values)
    ood_values = ood_values[finite_ood]
    sources = sources[finite_ood]
    if id_values.size == 0 or ood_values.size == 0:
        return metrics

    def compute_auroc(selected_ood: np.ndarray) -> float:
        targets = np.concatenate((np.zeros(id_values.size), np.ones(selected_ood.size)))
        scores = np.concatenate((id_values, selected_ood))
        return float(roc_auc_score(targets, scores))

    overall = compute_auroc(ood_values)
    metrics["ood/AUROC"] = overall
    metrics["ood/overall_AUROC"] = overall

    source_aurocs = []
    for source in pd.unique(sources):
        source_values = ood_values[sources == source]
        if source_values.size == 0:
            continue
        source_name = Path(source).stem
        source_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", source_name).strip("_")
        source_auroc = compute_auroc(source_values)
        metrics[f"ood/source_AUROC/{source_name}"] = source_auroc
        source_aurocs.append(source_auroc)

    if source_aurocs:
        metrics["ood/macro_AUROC"] = float(np.mean(source_aurocs))
    return metrics


def promote_postprocess_metrics(results: Results) -> Results:
    """Use calibrated classification metrics as the canonical table metrics.

    The raw base-model metrics remain available under ``base/...`` and the
    original ``test/post/...`` keys are retained for detailed inspection.
    Conformal-only metrics such as coverage and set size are left unchanged.
    """
    for metrics in results.values():
        for key, value in list(metrics.items()):
            if not key.startswith("test/post/"):
                continue
            suffix = key.removeprefix("test/post/")
            if not suffix.startswith(("cls/", "cal/", "sc/")):
                continue
            canonical = f"test/{suffix}"
            if canonical in metrics:
                metrics[f"base/{canonical}"] = metrics[canonical]
            metrics[canonical] = value
    return results


def save_prediction_artifacts(
    trainer: TUTrainer,
    routine,
    config: str,
    datamodule: Any | None = None,
) -> Path | None:
    """Save real per-sample predictions used by standalone plotting commands."""
    logger = getattr(trainer, "logger", None)
    save_dir = getattr(logger, "save_dir", None)
    if save_dir is None or not hasattr(routine, "get_prediction_artifacts"):
        return None
    tensors = routine.get_prediction_artifacts()
    if not tensors:
        return None
    output = Path(save_dir) / "predictions" / f"{config}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()}
    arrays["ood_criterion"] = np.asarray(type(routine.ood_criterion).__name__)
    ood_frame = getattr(datamodule, "ood_df", None)
    if (
        ood_frame is not None
        and "source" in ood_frame
        and "ood_scores" in arrays
        and len(ood_frame) == len(arrays["ood_scores"])
    ):
        # Force a fixed-width Unicode dtype. Pandas otherwise returns an
        # object array, which NumPy refuses to load with allow_pickle=False.
        arrays["ood_sources"] = np.asarray(
            ood_frame["source"].astype(str).tolist(), dtype=np.str_
        )
    np.savez_compressed(output, **arrays)
    return output


def add_experiment_args(parser, *, checkpoint: bool = False):
    """Add the repeatability arguments shared by every method script."""
    config = load_config(parser.get_default("config"))
    training = config.get("training", {})
    runner = config.get("runner", {})
    parser.add_argument("--n-runs", type=int, default=training.get("n_runs", 5))
    parser.add_argument("--seeds", type=int, nargs="+", default=training.get("seeds"))
    parser.add_argument("--val-split", type=float, default=training.get("val_split", 0.2))
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=runner.get("overwrite", True),
        help="Replace the existing result directory for this dataset/backbone/method",
    )
    if checkpoint:
        parser.add_argument("--ckpt", type=str, default=None)
    return parser


def method_dir(args, method: str) -> Path:
    return Path(args.output_dir) / str(args.dataset) / args.backbone / method


def seed_dir(args, method: str, seed: int) -> Path:
    return method_dir(args, method) / f"seed{seed}"


def make_trainer(
    args,
    run_dir: Path,
    *,
    max_epochs: int | None = None,
    monitor: str | None = "val/cls/NLL",
    mode: str = "min",
    callbacks: Sequence[Any] = (),
    checkpoint: bool = True,
    checkpoint_name: str = "best-{epoch}",
    **trainer_overrides,
) -> TUTrainer:
    """Create a consistently logged trainer and optional best checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    all_callbacks = [TQDMProgressBar(refresh_rate=1)]
    if checkpoint:
        all_callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir / "ckpt",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                filename=checkpoint_name,
                verbose=False,
            )
        )
    all_callbacks.extend(callbacks)
    kwargs = get_trainer_kwargs(args)
    kwargs.update(trainer_overrides)
    return TUTrainer(
        **kwargs,
        max_epochs=max_epochs if max_epochs is not None else args.epochs,
        logger=CSVLogger(save_dir=str(run_dir), name="", version="logs"),
        callbacks=all_callbacks,
    )


def load_best_weights(trainer: TUTrainer, routine) -> Path:
    """Load the best ModelCheckpoint weights into the fitted routine in place."""
    checkpoint = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    )
    path = Path(checkpoint.best_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Best checkpoint was not created: {path}")
    payload = torch.load(path, map_location="cpu")
    routine.load_state_dict(payload["state_dict"])
    return path


def train_base_classifier(
    args,
    run_dir: Path,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
):
    """Train a deterministic base classifier and restore its best weights."""
    datamodule = get_datamodule(args, val_split=args.val_split, eval_ood=True, eval_shift=args.eval_shift)
    model = get_model(
        args.backbone,
        datamodule.num_channels,
        datamodule.num_classes,
        **dict(model_kwargs or {}),
    )
    routine = ClassificationRoutine(
        model=model,
        num_classes=datamodule.num_classes,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
    )
    trainer = make_trainer(args, run_dir)
    trainer.fit(model=routine, datamodule=datamodule)
    load_best_weights(trainer, routine)
    return trainer, datamodule, model


def train_postprocess_and_evaluate(
    args,
    run_dir: Path,
    build_postprocess: Callable[[Any, Any], Any],
    *,
    ood_criterion: Any | None = None,
) -> Results:
    """Train a base classifier, fit one post-processor, then evaluate it."""
    trainer, datamodule, model = train_base_classifier(args, run_dir)
    postprocess = build_postprocess(model, datamodule)
    if getattr(postprocess, "model", None) is None:
        postprocess.set_model(model)
    postprocess.fit(datamodule.postprocess_dataloader())
    postprocess.fit = lambda *args, **kwargs: None
    routine = ClassificationRoutine(
        model=model,
        num_classes=datamodule.num_classes,
        loss=None,
        eval_ood=True, eval_shift=args.eval_shift,
        post_processing=postprocess,
        log_post_processing=True,
        ood_criterion=ood_criterion or "msp",
    )
    return promote_postprocess_metrics(evaluate(args, trainer, routine, datamodule))


def evaluate(
    args,
    trainer: TUTrainer,
    routine,
    datamodule,
    *,
    ckpt_path: str | Path | None = None,
    on_config: Callable[[str], None] | None = None,
    artifact_prefix: str = "",
) -> Results:
    """Evaluate clean, operating-shift, and configured noisy conditions."""
    results: Results = {}
    # Benchmark metrics are persisted once by ``run_repeated``. Some legacy
    # routines still request a second, incompatible per-test CSV artifact.
    routine.save_in_csv = False
    routine.collect_predictions = True

    shift_requested = bool(getattr(args, "eval_shift", False))
    datamodule_shift_state = bool(getattr(datamodule, "eval_shift", False))
    routine_shift_state = bool(getattr(routine, "eval_shift", False))
    if shift_requested and not (datamodule_shift_state and routine_shift_state):
        raise RuntimeError(
            "Shift evaluation was requested, but it was not enabled on both "
            "the data module and classification routine."
        )

    operating_shift_set = None
    if shift_requested:
        if not hasattr(datamodule, "get_shift_set"):
            raise NotImplementedError(
                f"{type(datamodule).__name__} does not support "
                "operating-condition shift evaluation."
            )
        # Build and cache the shift set before suppressing its auxiliary loader.
        datamodule.setup("test")
        operating_shift_set = datamodule.get_shift_set()

    # The operating-condition shift is evaluated once as the primary test set.
    # Suppress the auxiliary shift loader during clean and noise evaluations.
    datamodule.eval_shift = False
    routine.eval_shift = False
    try:
        if on_config:
            on_config("clean")
        clean_metrics = trainer.test(
            model=routine,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )[0]
        results["clean"] = add_ood_source_metrics(
            clean_metrics, routine, datamodule
        )
        save_prediction_artifacts(
            trainer, routine, f"{artifact_prefix}clean", datamodule
        )

        if shift_requested:
            config = "operating_shift"
            print(f"\n=== {config} ===")
            original_test = datamodule.test
            datamodule_ood_state = bool(getattr(datamodule, "eval_ood", False))
            routine_ood_state = bool(getattr(routine, "eval_ood", False))
            noisy_mode_state = bool(getattr(datamodule, "_noisy_mode", False))
            try:
                datamodule.test = operating_shift_set
                datamodule.eval_ood = False
                routine.eval_ood = False
                datamodule._noisy_mode = True
                if on_config:
                    on_config(config)
                shift_metrics = trainer.test(
                    model=routine,
                    datamodule=datamodule,
                )[0]
                results[config] = shift_metrics
                save_prediction_artifacts(
                    trainer,
                    routine,
                    f"{artifact_prefix}{config}",
                    datamodule,
                )
            finally:
                datamodule.test = original_test
                datamodule.eval_ood = datamodule_ood_state
                routine.eval_ood = routine_ood_state
                datamodule._noisy_mode = noisy_mode_state

        if not getattr(args, "eval_noise", False):
            return results

        required = ("noise_params", "get_noisy_test_set", "get_noisy_ood_set")
        missing = [name for name in required if not hasattr(datamodule, name)]
        if missing:
            raise NotImplementedError(
                f"{type(datamodule).__name__} does not support noise evaluation "
                f"({', '.join(missing)} missing). Run with --no-eval-noise."
            )

        for noise_type in datamodule.noise_params:
            for severity in range(1, 6):
                config = f"{noise_type}_s{severity}"
                print(f"\n=== {config} ===")
                datamodule.test = datamodule.get_noisy_test_set(
                    noise_type, severity
                )
                if datamodule.eval_ood:
                    datamodule.ood = datamodule.get_noisy_ood_set(
                        noise_type, severity
                    )
                datamodule._noisy_mode = True
                try:
                    if on_config:
                        on_config(config)
                    noisy_metrics = trainer.test(
                        model=routine,
                        datamodule=datamodule,
                    )[0]
                    results[config] = add_ood_source_metrics(
                        noisy_metrics, routine, datamodule
                    )
                    save_prediction_artifacts(
                        trainer,
                        routine,
                        f"{artifact_prefix}{config}",
                        datamodule,
                    )
                finally:
                    datamodule._noisy_mode = False
        return results
    finally:
        datamodule.eval_shift = datamodule_shift_state
        routine.eval_shift = routine_shift_state
def fit_and_evaluate(
    args,
    run_dir: Path,
    build_routine: Callable[[Any], Any],
    *,
    monitor: str = "val/cls/NLL",
    mode: str = "min",
    callbacks: Sequence[Any] = (),
    trainer_overrides: Mapping[str, Any] | None = None,
) -> Results:
    """Run the standard train-best-checkpoint-clean/noise workflow."""
    datamodule = get_datamodule(
        args,
        val_split=args.val_split,
        eval_ood=True, eval_shift=args.eval_shift,
    )
    routine = build_routine(datamodule)
    trainer = make_trainer(
        args,
        run_dir,
        monitor=monitor,
        mode=mode,
        callbacks=callbacks,
        **dict(trainer_overrides or {}),
    )
    trainer.fit(model=routine, datamodule=datamodule)
    return evaluate(args, trainer, routine, datamodule, ckpt_path="best")


def save_seed_metrics(run_dir: Path, seed: int, results: Mapping[str, Metrics]) -> None:
    rows = [
        {"seed": seed, "config": config, **metrics}
        for config, metrics in results.items()
    ]
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, tidy summary instead of pandas' two-row CSV header."""
    metric_cols = [
        col for col in frame.columns
        if col not in {"seed", "config"} and pd.api.types.is_numeric_dtype(frame[col])
    ]
    rows: list[dict[str, Any]] = []
    for config, group in frame.groupby("config", sort=False):
        if str(config).startswith(("baseline_", "before_")):
            continue
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "config": config,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            })
    return pd.DataFrame(rows, columns=("config", "metric", "mean", "std", "n"))


def _write_manifest(
    out_dir: Path,
    args: Any,
    method: str,
    seeds: Sequence[int],
    *,
    status: str,
) -> None:
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "dataset": str(args.dataset),
        "backbone": str(args.backbone),
        "method": method,
        "seeds": list(seeds),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "runs": "runs.csv",
            "summary": "summary.csv",
            "seed_metrics": "seed<N>/metrics.csv",
            "predictions": "seed<N>/predictions/<config>.npz",
        },
        "arguments": vars(args),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def run_repeated(
    args,
    method: str,
    run_once: Callable[[Any, int, Path], Results],
) -> pd.DataFrame:
    """Run all seeds, persist partial results, and produce mean/std summaries."""
    seeds = args.seeds if args.seeds else list(range(args.n_runs))
    out_dir = method_dir(args, method)
    if args.overwrite and out_dir.exists():
        output_root = Path(args.output_dir).resolve()
        target = out_dir.resolve()
        if target == output_root or not target.is_relative_to(output_root):
            raise ValueError(f"Refusing to remove unsafe experiment directory: {target}")
        shutil.rmtree(target)
        print(f"Removed previous results: {target}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(out_dir, args, method, seeds, status="running")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n{'=' * 60}\n  {method} | seed={seed}\n{'=' * 60}")
        seed_everything(seed, workers=True)
        run_dir = seed_dir(args, method, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        results = run_once(args, seed, run_dir)
        save_seed_metrics(run_dir, seed, results)
        rows.extend(
            {"seed": seed, "config": config, **metrics}
            for config, metrics in results.items()
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "runs.csv", index=False)
    _build_summary(frame).to_csv(out_dir / "summary.csv", index=False)
    _write_manifest(out_dir, args, method, seeds, status="complete")

    print(f"\nSaved results to {out_dir}")
    clean = frame[frame["config"] == "clean"]
    if clean.empty and method == "temperature_scaling":
        clean = frame[frame["config"] == "after_clean"]
    source_auroc_cols = sorted(
        col for col in clean.columns if col.startswith("ood/source_AUROC/")
    )
    preferred = (
        "seed",
        "test/cls/Acc",
        "test/cal/ECE",
        "ood/overall_AUROC",
        *source_auroc_cols,
        "ood/macro_AUROC",
        "ood/FPR95",
    )
    display = [col for col in preferred if col in clean.columns]
    if display:
        print(clean[display].to_string(index=False))
    return frame

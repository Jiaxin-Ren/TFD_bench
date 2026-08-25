import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn, optim

from src.metrics.ood import MutualInformationCriterion
from src.models.wrappers import CheckpointCollector
from src.optim import SGHMC
from src.training import ClassificationRoutine
from src.training.experiment import (
    add_experiment_args,
    evaluate,
    load_best_weights,
    make_trainer,
    require_ensemble_samples,
    run_repeated,
)
from src.utils import add_common_args, get_datamodule, get_model, load_gpu_config, print_gpu_config

METHOD_NAME = "sghmc"


def planned_collection_size(
    epochs: int,
    cycle_start: int,
    cycle_length: int,
    num_estimators: int,
) -> int:
    if epochs < 1 or cycle_start < 0:
        raise ValueError("epochs must be positive and cycle_start non-negative.")
    if cycle_length < 1:
        raise ValueError("cycle_length must be positive.")
    if num_estimators < 2:
        raise ValueError("num_estimators must be at least 2.")
    available = 1 + len(range(cycle_start, epochs, cycle_length))
    return min(num_estimators, available)


def validate_collection_schedule(args) -> int:
    planned = planned_collection_size(
        args.epochs,
        args.cycle_start,
        args.cycle_length,
        args.num_estimators,
    )
    if planned < args.num_estimators:
        raise ValueError(
            f"SGHMC can collect only {planned} estimators with epochs={args.epochs}, "
            f"cycle_start={args.cycle_start}, and cycle_length={args.cycle_length}; "
            f"{args.num_estimators} were requested."
        )
    return planned


def run_once(args, seed, run_dir):
    validate_collection_schedule(args)
    dm = get_datamodule(args, val_split=args.val_split, eval_ood=True, eval_shift=args.eval_shift)
    base = get_model(args.backbone, dm.num_channels, dm.num_classes)
    pretrain = ClassificationRoutine(
        model=base, num_classes=dm.num_classes, loss=nn.CrossEntropyLoss(),
        optim_recipe=optim.SGD(base.parameters(), lr=args.pretrain_lr,
                               momentum=0.9, weight_decay=1e-4),
    )
    pretrainer = make_trainer(args, run_dir / "pretrain", max_epochs=args.pretrain_epochs)
    pretrainer.fit(pretrain, datamodule=dm)
    load_best_weights(pretrainer, pretrain)

    model = CheckpointCollector(
        base, cycle_start=args.cycle_start, cycle_length=args.cycle_length,
        use_final_model=True, max_num_estimators=args.num_estimators,
    )
    routine = ClassificationRoutine(
        model=model, num_classes=dm.num_classes, is_ensemble=True,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=SGHMC(
            model.parameters(), lr=args.lr, friction=args.friction,
            burn_in_steps=args.burn_in_steps, noise_factor=args.noise_factor,
            weight_decay=1e-3,
        ),
        eval_ood=True, eval_shift=args.eval_shift, ood_criterion=MutualInformationCriterion(),
    )
    trainer = make_trainer(args, run_dir, checkpoint=False)
    trainer.fit(routine, datamodule=dm)
    require_ensemble_samples(model, METHOD_NAME, args.num_estimators)
    return evaluate(args, trainer, routine, dm, ckpt_path=None)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--cycle-start", type=int, default=10)
    parser.add_argument("--cycle-length", type=int, default=4)
    parser.add_argument("--friction", type=float, default=0.05)
    parser.add_argument("--burn-in-steps", type=int, default=200)
    parser.add_argument("--noise-factor", type=float, default=1e-2)
    parser.add_argument("--num-estimators", type=int, default=10)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)

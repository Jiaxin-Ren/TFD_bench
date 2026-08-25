import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn

from src.metrics.ood import MutualInformationCriterion
from src.post_processing.mc_batch_norm import MCBatchNorm
from src.training import ClassificationRoutine
from src.training.experiment import (
    add_experiment_args,
    evaluate,
    run_repeated,
    train_base_classifier,
)
from src.utils import add_common_args, load_gpu_config, print_gpu_config

METHOD_NAME = "mc_batch_norm"


def run_once(args, seed, run_dir):
    trainer, datamodule, model = train_base_classifier(
        args, run_dir,
        model_kwargs={"norm": nn.BatchNorm1d} if args.backbone == "resnet" else None,
    )
    mc_model = MCBatchNorm(
        model, num_estimators=args.num_estimators, convert=True,
        mc_batch_size=args.mc_batch_size, last_only=args.last_only,
    )
    mc_model.fit(datamodule.train_dataloader())
    routine = ClassificationRoutine(
        model=mc_model, num_classes=datamodule.num_classes,
        loss=nn.CrossEntropyLoss(), is_ensemble=True, eval_ood=True, eval_shift=args.eval_shift,
        ood_criterion=MutualInformationCriterion(),
    )
    return evaluate(args, trainer, routine, datamodule)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--mc-batch-size", type=int, default=4)
    parser.add_argument("--last-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn, optim

from src.metrics.ood import MutualInformationCriterion
from src.models.wrappers import SWAG
from src.training import ClassificationRoutine
from src.training.experiment import (
    add_experiment_args,
    evaluate,
    make_trainer,
    require_ensemble_samples,
    run_repeated,
)
from src.utils import (
    add_common_args,
    get_datamodule,
    get_model,
    load_gpu_config,
    print_gpu_config,
)

METHOD_NAME = "swag"


def run_once(args, seed, run_dir):
    dm = get_datamodule(args, val_split=args.val_split, eval_ood=True, eval_shift=args.eval_shift)
    base = get_model(args.backbone, dm.num_channels, dm.num_classes)
    model = SWAG(
        base,
        cycle_start=args.cycle_start,
        cycle_length=args.cycle_length,
        num_estimators=args.num_estimators,
    )
    routine = ClassificationRoutine(
        model=model, num_classes=dm.num_classes, is_ensemble=True,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
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
    parser.add_argument("--cycle-start", type=int, default=25)
    parser.add_argument("--cycle-length", type=int, default=1)
    parser.add_argument("--num-estimators", type=int, default=16)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)

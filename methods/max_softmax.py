import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn, optim

from src.metrics.ood import MaxSoftmaxCriterion
from src.training import ClassificationRoutine
from src.training.experiment import (
    add_experiment_args,
    evaluate,
    make_trainer,
    run_repeated,
)
from src.utils import add_common_args, get_datamodule, get_model, load_gpu_config, print_gpu_config

METHOD_NAME = "max_softmax"


def run_once(args, seed, run_dir):
    datamodule = get_datamodule(args, val_split=args.val_split, eval_ood=True, eval_shift=args.eval_shift)
    model = get_model(args.backbone, datamodule.num_channels, datamodule.num_classes)
    routine = ClassificationRoutine(
        model=model, num_classes=datamodule.num_classes,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
        eval_ood=True, eval_shift=args.eval_shift, ood_criterion=MaxSoftmaxCriterion(),
    )
    trainer = make_trainer(args, run_dir)
    if args.ckpt is None:
        trainer.fit(model=routine, datamodule=datamodule)
        checkpoint = "best"
    else:
        checkpoint = args.ckpt
    return evaluate(args, trainer, routine, datamodule, ckpt_path=checkpoint)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()), checkpoint=True)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)

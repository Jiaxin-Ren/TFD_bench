# Reproducibility Guide

This document defines the minimum procedure for reproducing TFD-Bench experiments.

## 1. Environment

TFD-Bench is validated with Python 3.10. Create an isolated environment and install the pinned runtime:

```bash
conda create -n tfd python=3.10 -y
conda activate tfd
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

Record the following with every published result:

```bash
python --version
python -c "import torch, lightning; print(torch.__version__, lightning.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Exact floating-point results can vary across PyTorch, CUDA, cuDNN, GPU models and distributed strategies. Compare the reported mean and standard deviation across the configured seeds rather than requiring bitwise equality.

## 2. Data

Datasets are not redistributed with this repository. Prepare them according to [DATASETS.md](DATASETS.md) and keep them below `data/` or another local directory ignored by Git.

Update every `datasets[].root` in the experiment YAML. A missing or incorrectly structured dataset must be treated as a setup error, not as an empty benchmark result.

## 3. Validate the installation

The following checks require no dataset:

```bash
python -m compileall -q run.py methods src analysis
python -m unittest discover -s tests -v
python run.py --dry-run
```

A one-seed CPU smoke experiment, after configuring the dataset path, is:

```bash
python methods/max_softmax.py \
  --dataset seu \
  --data-root ./data/SEU \
  --backbone resnet \
  --epochs 1 \
  --seeds 0 \
  --accelerator cpu \
  --devices 1 \
  --no-eval-noise
```

This smoke run checks the data loader, training loop, checkpoint selection, clean ID/OOD evaluation and schema-v2 result writer. It is not a benchmark result.

## 4. Run the benchmark

Copy the configuration before changing it so the exact experiment definition can be archived:

```bash
cp configs/default.yaml configs/my_experiment.yaml
python run.py --config configs/my_experiment.yaml --dry-run
python run.py --config configs/my_experiment.yaml
```

For a fair method comparison, keep these fields identical across methods:

- dataset roots and ID/OOD definitions;
- backbone and input preprocessing;
- training epochs, batch size and validation split;
- seed list;
- ensemble/posterior estimator count where applicable;
- noise types and severities.

Normal methods select the checkpoint with minimum validation NLL. SGLD, SGHMC and SWAG evaluate the complete posterior sample collection produced after their collection schedule; their deterministic pretraining checkpoint is still selected by validation NLL. Test and OOD metrics never select checkpoints.

## 5. Result contract

Each method writes:

```text
results/<dataset>/<backbone>/<method>/
├── manifest.json
├── runs.csv
├── summary.csv
└── seed<N>/
    ├── metrics.csv
    ├── predictions/<config>.npz
    ├── ckpt/
    └── logs/
```

`manifest.json` records schema version, arguments, seeds and completion status. `runs.csv` is the canonical method-level raw table. `summary.csv` is a tidy mean/std/count table. Prediction archives retain per-sample probabilities, labels and uncertainty scores for independent numerical verification.

Primary comparison metrics are ACC, ECE and OOD AUROC. OOD output also includes overall AUROC, per-source AUROC and macro AUROC. Report the configured seed count and do not silently combine results produced with different configurations.

## 6. Collect results and generate tables

```bash
python analysis/collect_results.py
python analysis/generate_tables.py
```

This creates:

```text
results/summary.json
results/tables/table.md
```

Result collection and table generation are separate from training and do not overwrite method checkpoints.

## 7. Archive a reproducible run

For a paper or release, archive:

- the Git commit hash;
- the exact YAML configuration;
- `manifest.json`, `runs.csv` and `summary.csv` for every method;
- `results/summary.json` and tables;
- Python/PyTorch/CUDA/GPU information;
- dataset version or download date and the original dataset citation.

Do not commit datasets, checkpoints or `results/` to this repository.

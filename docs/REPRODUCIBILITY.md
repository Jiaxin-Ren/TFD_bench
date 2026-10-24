# Reproducibility Guide

This document describes how to run the experiments accompanying the MSSP
manuscript. The formal evaluation covers six datasets (SEU, WT, PU, XJTU, HIT and
CWRU), fourteen methods including MSP, and a shared one-dimensional ResNet backbone.
MGB and THU are additional supported loaders, outside this six-dataset evaluation.

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
  --no-eval-noise \
  --no-eval-shift \
  --output-dir ./tmp/smoke_results
```

This smoke run checks the data loader, training loop, checkpoint selection, clean ID/OOD evaluation and schema-v2 result writer. Its output is separate from the formal results in `results/`; it is not a benchmark result.

## 4. Run the benchmark

The matrix runner and shared arguments default to `configs/default.yaml`.
Standalone methods take method-specific defaults directly from their scripts,
including any dataset-specific defaults defined there; explicit command-line
options override them. The matrix runner applies YAML `method_args` and then
`method_dataset_args` as command-line options, so dataset-specific YAML values
take precedence over shared method values. Standalone methods do not automatically
apply those two YAML sections. Copy the configuration before changing it so the
exact experiment definition can be archived:

```bash
cp configs/default.yaml configs/my_experiment.yaml
python run.py --config configs/my_experiment.yaml --dry-run
python run.py --config configs/my_experiment.yaml
```

The shared training configuration specifies 50 epochs and seeds 0 through 4.
SGLD and SGHMC use an additional 20-epoch deterministic pretraining stage followed
by 50 sampling-stage epochs. SWAG collects its posterior approximation during its
50-epoch training run and has no separate deterministic pretraining stage. An
alternative YAML can be selected explicitly, for example
`python run.py --config configs/tuning.yaml`.

By default, `runner.overwrite: true` replaces the selected
`results/<dataset>/<backbone>/<method>/` directories. To keep previous experiments,
choose a separate `output.dir` before running. Setting `runner.overwrite: false`
disables the initial directory removal; it does not create a separate experiment
archive or guarantee that same-named output files will be preserved.

For a fair method comparison, keep these fields identical across methods:

- dataset roots and ID/OOD definitions;
- backbone and input preprocessing;
- shared training settings, batch size and validation split;
- seed list;
- noise types and severities.

Keep each method's documented training stages and estimator budget. Estimator
counts are method-specific: for example, DE, PE and BE use four members, whereas
several posterior and Monte Carlo methods use ten samples. They are not required
to be identical across all methods.

Conventional methods select the checkpoint with minimum validation NLL. SGLD and
SGHMC also select their deterministic pretraining checkpoint by validation NLL,
then evaluate the posterior sample collection produced by the sampling stage.
SWAG evaluates its collected posterior approximation. Test and OOD metrics never
select checkpoints.

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
    ├── ckpt/                  # Validation-selected checkpoint, where applicable
    └── logs/                  # Training logs; sampler checkpoints in checkpoints/
```

`manifest.json` records schema version, arguments, seeds and completion status. `runs.csv` is the canonical method-level raw table. `summary.csv` is a tidy mean/std/count table. Prediction archives retain per-sample probabilities, labels and uncertainty scores for independent numerical verification.

Prediction configurations include `clean`, `gaussian_s1` through `gaussian_s5`
(14, 12, 10, 8 and 6 dB), and `operating_shift`. XJTU's `operating_shift` archive
represents a sensor-position shift; the other five datasets use conventional
operating-condition shifts. Temperature scaling retains both `baseline_<config>`
and `after_<config>` outputs.

The paper evaluates four dimensions: closed-set classification (ACC), confidence
calibration (ECE), OOD detection (AUROC), and selective classification (AURC).
NLL and the Brier score complement these with measures of predictive probability
quality. All six metrics are retained for clean and Gaussian-noise evaluation.
The shift evaluation retains known fault classes and does not compute OOD AUROC.
Additional OOD outputs include per-source and macro AUROC. Report the configured
seed count and do not silently combine results produced with different configurations.

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

The collector retains all available metrics. The table generator displays only
ACC, ECE and AUROC by default; this display choice does not limit the training
outputs. To show all six metrics:

```bash
python analysis/generate_tables.py --metrics test/cls/Acc test/cls/NLL test/cls/Brier test/cal/ECE ood/AUROC test/sc/AURC
```

## 7. Archive a reproducible run

For a paper or release, archive:

- the Git commit hash;
- the exact YAML configuration;
- `manifest.json`, `runs.csv` and `summary.csv` for every method;
- `results/summary.json` and tables;
- Python/PyTorch/CUDA/GPU information;
- dataset version or download date and the original dataset citation.

Do not commit datasets, checkpoints or `results/` to this repository.

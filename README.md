# Beyond classification accuracy: Benchmarking uncertainty quantification for trustworthy mechanical fault diagnosis

**TFD-Bench** is the research code accompanying the manuscript above, prepared for
*Mechanical Systems and Signal Processing* (MSSP). It benchmarks uncertainty
quantification (UQ) for mechanical fault diagnosis beyond classification
accuracy: whether predictive confidence is reliable, unknown faults can be
identified, and uncertain predictions can be deferred for further inspection.

The repository provides independent method implementations, a shared experiment
runner, and a common result format for reproducible comparison.

## Benchmark overview

- **6 vibration datasets:** SEU, WT, PU, XJTU, HIT, and CWRU.
- **14 evaluated methods:** thirteen UQ methods from five families, plus the MSP reference.
- **4 evaluation dimensions:** classification, confidence calibration, OOD detection,
  and selective classification.
- **Realistic test conditions:** clean inputs, Gaussian noise, acquisition/operating-condition
  shifts, and unknown fault classes.
- **Shared experimental setup:** a 1D ResNet, fixed data partitions, and five training seeds.
- **Reusable outputs:** per-seed metrics, statistical summaries, and per-sample predictions.

The paper's six-dataset, common-ResNet comparison is the formal benchmark.
Additional backbones, MGB/THU loaders, and Checkpoint Ensemble are retained for
extension but are not part of that comparison.

## Methods

The families and abbreviations follow the manuscript. Maximum Softmax Probability
(MSP) is a deterministic reference, separate from the five UQ families.

| Family | Method | Abbreviation | Entry point |
|---|---|---|---|
| Reference | Maximum Softmax Probability | MSP | [max_softmax.py](methods/max_softmax.py) |
| Post-hoc calibration | Temperature Scaling | TS | [temperature_scaling.py](methods/temperature_scaling.py) |
| Bayesian inference | Variational Bayesian Neural Network | VBNN | [variational_bnn.py](methods/variational_bnn.py) |
| Bayesian inference | Stochastic Gradient Langevin Dynamics | SGLD | [sgld.py](methods/sgld.py) |
| Bayesian inference | Stochastic Gradient Hamiltonian Monte Carlo | SGHMC | [sghmc.py](methods/sghmc.py) |
| Ensemble | Deep Ensemble | DE | [deep_ensemble.py](methods/deep_ensemble.py) |
| Ensemble | Snapshot Ensemble | SE | [snapshot_ensemble.py](methods/snapshot_ensemble.py) |
| Ensemble | Batch Ensemble | BE | [batch_ensemble.py](methods/batch_ensemble.py) |
| Ensemble | Packed Ensemble | PE | [packed_ensemble.py](methods/packed_ensemble.py) |
| Posterior approximation | Stochastic Weight Averaging Gaussian | SWAG | [swag.py](methods/swag.py) |
| Posterior approximation | Laplace Approximation | LA | [laplace_approx.py](methods/laplace_approx.py) |
| Posterior approximation | Monte Carlo Dropout | MCD | [mc_dropout.py](methods/mc_dropout.py) |
| Posterior approximation | Monte Carlo Batch Normalization | MCBN | [mc_batch_norm.py](methods/mc_batch_norm.py) |
| Evidential learning | Evidential Deep Learning | EDL | [edl.py](methods/edl.py) |

[Checkpoint Ensemble](methods/checkpoint_ensemble.py) is an additional implementation
and is excluded from standard benchmark collection and comparison tables.

## Evaluation

| Dimension | Primary metric | What is evaluated? |
|---|---|---|
| Classification | ACC ↑ | Correct diagnosis of known fault classes |
| Confidence calibration | ECE ↓ | Agreement between confidence and empirical correctness |
| OOD detection | AUROC ↑ | Separation of known-class inputs and unknown faults |
| Selective classification | AURC ↓ | Error risk as uncertain predictions are deferred |

NLL and the Brier score complement these metrics by measuring predictive
probability quality. ECE uses **10 bins**. Detection and selective classification
use method-native uncertainty scores, oriented so that larger scores indicate
greater uncertainty.

Evaluation includes:

- **Clean ID:** held-out samples from known fault classes.
- **Gaussian-noise shift:** five test-time SNRs—14, 12, 10, 8, and 6 dB—applied
  to both ID and OOD signals.
- **Acquisition/operating-condition shift:** changed acquisition conditions with
  the known class space retained. XJTU uses a sensor-position shift and is
  reported separately from the other five datasets.
- **Unknown-fault OOD:** fault classes excluded from training.

The standalone shift configuration evaluates classification, calibration, and
selective classification; it does not compute OOD detection metrics.

## Datasets

Download data from the original public releases and follow
[the dataset preparation guide](docs/DATASETS.md) for the loader-compatible layout.

| Dataset | Public source |
|---|---|
| SEU | [Southeast University mechanical datasets](https://github.com/cathysiyu/Mechanical-datasets) |
| WT | [Wind-turbine planetary gearbox dataset](https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset) |
| PU | [Paderborn University Bearing Data Center](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download) |
| XJTU | [XJTUGearbox: authors' data release](https://github.com/HazeDT/PHMGNNBenchmark#datasets) |
| HIT | [HIT inter-shaft bearing dataset](https://github.com/HouLeiHIT/HIT-dataset) |
| CWRU | [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) |

Set `datasets[].root` to your local paths before running. XJTU refers to the
gearbox dataset above, not XJTU-SY. The HIT loader expects `data1.npy` through
`data5.npy`; the author's single-channel MATLAB demo alone is not sufficient.

Datasets are not redistributed here and retain their original access and license
terms. Parsed signals are cached automatically; use `TFD_DATA_CACHE` to change the
default cache root, `~/.cache/tfd_bench/datasets/`.

## Installation

Use **Python 3.10**. From the repository root:

~~~bash
conda create -n tfd python=3.10 -y
conda activate tfd
pip install -r requirements.txt
python -m pip check
~~~

The pinned runtime includes PyTorch, Lightning, and the dependencies required by
the supported methods, including Laplace Approximation.

## Quick start

### 1. Verify the installation

After preparing SEU, run a one-epoch smoke test in a separate output directory:

~~~bash
python methods/max_softmax.py --dataset seu --data-root ./data/SEU --epochs 1 --seeds 0 --no-eval-noise --no-eval-shift --output-dir tmp/smoke_results
~~~

This checks the data and training pipeline; it is not a benchmark result.

### 2. Run the benchmark

The selected method parameters are provided in
[configs/tuning.yaml](configs/tuning.yaml). Update the dataset paths, then preview
and run the experiment matrix:

~~~bash
python run.py --config configs/tuning.yaml --dry-run
python run.py --config configs/tuning.yaml
~~~

The configuration schedules **84 method–dataset combinations**, each with seeds
0–4. A dry run prints the commands without training or checking dataset contents.

To preserve the supplied configuration, copy it to a new YAML before customizing
it. Change `output.dir` for a new experiment: the default
`runner.overwrite: true` replaces existing results for selected combinations.

### 3. Run an individual method

Each method is independently executable:

~~~bash
python methods/edl.py --dataset seu --data-root ./data/SEU --backbone resnet
python methods/edl.py --help
~~~

Shared arguments come from [configs/default.yaml](configs/default.yaml), or the
file selected with `--config`. Method-specific defaults are defined directly in
the method script. Explicit command-line arguments override these defaults.

### 4. Run independent jobs on multiple GPUs

~~~bash
python run.py --config configs/tuning.yaml --workers 4 --gpus 0,1,2,3
~~~

Each worker runs an independent single-GPU experiment. This is not multi-GPU
training of one model. The runner saves job status to
`<output.dir>/runner_status.csv` and process logs to `<output.dir>/_runner_logs/`.

## Configuration

| Configuration field | Purpose |
|---|---|
| `datasets`, `backbones`, `methods` | Select the experiment matrix and dataset paths |
| `training` | Shared epochs, batch size, learning rate, validation ratio, and seeds |
| `evaluation.noise`, `evaluation.shift` | Enable noise and shift evaluations |
| `method_args` | Method-specific parameters |
| `method_dataset_args` | Method-specific overrides for individual datasets |
| `hardware`, `output`, `runner` | Device settings, output directory, and job execution |

For `run.py`, method settings follow:
**script defaults → `method_args` → `method_dataset_args`**.
The runner passes YAML settings as command-line arguments.
Standalone scripts do **not** automatically apply those two YAML sections.

The shared setup uses batch size 128, learning rate 0.001, validation ratio 0.2,
and 32-bit precision. The common epoch setting is 50; SGLD and SGHMC additionally
use 20 pretraining epochs. Sampling and ensemble budgets remain method-specific.
See [the reproducibility guide](docs/REPRODUCIBILITY.md) for training stages,
checkpoint selection, and parameter handling.

Running `python run.py` without `--config` uses `configs/default.yaml`.

## Results and analysis

Each method saves:

~~~text
results/<dataset>/<backbone>/<method>/
├── manifest.json          # Arguments, seeds, run status, and schema version
├── runs.csv               # Per-seed/per-condition metrics
├── summary.csv            # Metric mean, sample standard deviation, and count
└── seed<N>/
    ├── metrics.csv
    ├── predictions/       # Probabilities, labels, and native uncertainty scores
    ├── ckpt/              # Where best-checkpoint saving is enabled
    └── logs/
~~~

Prediction archives cover enabled conditions such as `clean`, `gaussian_s1`
through `gaussian_s5`, and `operating_shift`. TS retains both uncalibrated
`baseline_*` and calibrated `after_*` outputs; standard TS summaries use only
the calibrated results.

Collect results and generate tables after training:

~~~bash
python analysis/collect_results.py
python analysis/generate_tables.py
~~~

Default outputs are `results/summary.json` and `results/tables/table.md`.
The collector includes all fourteen methods. The convenience table displays the
thirteen UQ methods with ACC, ECE, and AUROC; MSP remains in the collected results.
To include all six paper metrics:

~~~bash
python analysis/generate_tables.py --metrics test/cls/Acc test/cls/NLL test/cls/Brier test/cal/ECE ood/AUROC test/sc/AURC
~~~

For a custom results directory:

~~~bash
python analysis/collect_results.py --results-dir results/my_run --output results/my_run/summary.json
python analysis/generate_tables.py --input results/my_run/summary.json --output results/my_run/table.md
~~~

Tables are grouped by dataset, backbone, and evaluation condition. These commands
do not automatically generate the paper's cross-dataset macro averages or figures.
The table utility displays ACC/ECE/AUROC as percentages and AURC on its original
scale; the paper presents AURC multiplied by 100.

## Repository and documentation

~~~text
configs/          Benchmark configurations
methods/          Independent method entry points
src/              Data loaders, models, losses, metrics, and training
analysis/         Numerical result collection and table generation
docs/             Dataset preparation and reproducibility guides
tests/            Configuration, data-split, cache, and runtime tests
run.py            Experiment matrix runner
requirements.txt  Runtime dependencies
~~~

- [Dataset preparation](docs/DATASETS.md): public sources, file layouts, and splits.
- [Reproducibility guide](docs/REPRODUCIBILITY.md): protocol details and result archiving.

Run the data-free checks with:

~~~bash
python -m unittest discover -s tests -v
python run.py --dry-run
~~~

## License and acknowledgements

Code is released under the [Apache License 2.0](LICENSE).
Third-party attribution is provided in [NOTICE](NOTICE). Portions of the UQ
components are adapted from
[TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty)
for one-dimensional mechanical fault diagnosis.

Please cite the accompanying manuscript, the original methods, and the dataset
providers when using this benchmark. The code license does not cover the datasets.

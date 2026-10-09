# TFD-Bench: Uncertainty Benchmark for Fault Diagnosis

面向一维故障诊断信号的不确定性量化基准。项目提供统一的数据加载、训练、OOD 检测、噪声鲁棒性评估和多随机种子统计流程。

## 功能

- 使用一个 YAML 配置批量运行 `datasets × methods × backbones`
- 每种方法也可以通过命令行独立运行
- 统一保存 checkpoint、逐次实验指标和均值/标准差
- 支持 ID、OOD 和多等级噪声评估
- 支持确定性、集成、贝叶斯、证据学习和后处理方法

## 项目结构

```text
.
├── configs/default.yaml    # 原始默认配置
├── configs/tuning.yaml     # 已调参的正式实验配置
├── methods/                # 每种不确定性方法的独立入口
├── src/                    # 数据、模型、损失、指标和训练核心
├── analysis/               # 数值结果汇总和表格生成
├── docs/                   # 数据准备与完整复现指南
├── tests/                  # 配置、结果 schema 与加载 smoke tests
├── requirements.txt        # Python 依赖及验证版本
├── run.py                  # 批量实验入口
├── LICENSE                 # Apache License 2.0
└── results/                # 默认输出目录（不进入 Git）
```

## 环境

建议使用 Python 3.10。以 Conda 为例：

```bash
conda create -n tfd python=3.10 -y
conda activate tfd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

安装完成后可以检查主要依赖：

```bash
python -c "import torch, lightning, torchmetrics; print(torch.__version__)"
```

[`requirements.txt`](requirements.txt) 的核心依赖版本来自当前 `fd` 环境，
其中 `laplace-torch` 供 `laplace_approx` 方法使用。

## 快速开始

### 方式一：通过总配置一键运行

修改 [`configs/default.yaml`](configs/default.yaml)：

```yaml
methods:
  - max_softmax
  - edl
  - mc_dropout

backbones:
  - resnet
  - transformer

datasets:
  - name: seu
    root: ./data/SEU
  - name: mgb
    root: ./data/MGB

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  seeds: [0, 1, 2]
  val_split: 0.2

evaluation:
  noise: true

hardware:
  accelerator: auto
  devices: 1
  strategy: auto
  precision: 32

output:
  dir: results
```

先预览将要执行的命令，不启动训练：

```bash
python run.py --dry-run
```

运行配置中的全部组合：

```bash
python run.py
```

默认会覆盖所选数据集、模型和方法对应的旧实验目录；不会删除其他未选中的实验结果。可在 `runner` 配置中通过 `overwrite` 关闭覆盖行为。训练完成后，再按“结果分析”一节中的命令生成数值汇总和表格。

`runner.workers` 未设置或为 `1` 时，`run.py` 会依次执行每个
`dataset × method × backbone` 组合。服务器有多张 GPU 时，可以让每张 GPU
运行一个独立的单卡任务：

```yaml
hardware:
  accelerator: gpu
  devices: 1
  strategy: auto

runner:
  workers: 8
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  resume: true
  continue_on_error: true
```

并行 runner 会为每个 worker 设置对应的 `CUDA_VISIBLE_DEVICES`，并强制子任务
使用 `devices=1`，不会让一个实验跨多张卡运行 DDP。运行状态保存在输出目录的
`runner_status.csv`，各任务日志保存在 `_runner_logs/`。`resume: true` 会跳过
manifest、`runs.csv` 和 `summary.csv` 均完整的组合。也可以通过
`--workers 4 --gpus 0,1,2,3 --resume` 临时覆盖 runner 配置。

默认在某个组合失败后继续运行其他组合，并在结束时汇总失败项；可通过配置中的
`runner.continue_on_error` 修改该行为。

> 并非所有方法都支持所有 backbone。`resnet` 的兼容性最完整；增加其他 backbone 前建议先用 `--dry-run` 检查实验矩阵，并单独运行一次对应方法。

### 方式二：使用另一份配置

复制默认配置并修改：

```bash
cp configs/default.yaml configs/my_experiment.yaml
python run.py --config configs/my_experiment.yaml
```

Windows PowerShell：

```powershell
Copy-Item configs/default.yaml configs/my_experiment.yaml
python run.py --config configs/my_experiment.yaml
```

### 方式三：直接运行单个 method

每个 `methods/*.py` 都是独立入口。它会读取默认配置，再使用命令行参数覆盖配置值。

使用默认配置运行 EDL：

```bash
python methods/edl.py
```

指定配置和 backbone：

```bash
python methods/edl.py \
  --config configs/default.yaml \
  --backbone transformer
```

完全通过命令行覆盖常用参数：

```bash
python methods/edl.py \
  --dataset mgb \
  --data-root ./data/MGB \
  --backbone resnet \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --seeds 0 1 2 \
  --val-split 0.2 \
  --reg-weight 0.01 \
  --loss-type digamma \
  --no-eval-noise
```

查看某个方法支持的全部参数：

```bash
python methods/edl.py --help
```

### 方式四：在 YAML 中设置方法专属参数

方法专属参数放在 `method_args` 下，只会传给对应方法：

```yaml
method_args:
  deep_ensemble:
    num_estimators: 4
    ood_criterion: mi

  packed_ensemble:
    num_estimators: 4
    alpha: 4
    gamma: 1

  edl:
    reg_weight: 0.01
    loss_type: digamma

  mc_dropout:
    dropout_rate: 0.1
    num_estimators: 50
    ood_criterion: mi

```

YAML 中的下划线参数会自动转换为命令行的连字符形式，例如 `num_estimators` 会转换为 `--num-estimators`。布尔值会转换为 `--flag` 或 `--no-flag`。

同一方法需要针对不同数据集使用不同的独有超参数时，使用
`method_dataset_args`。第一层是方法名，第二层是数据集名：

```yaml
method_args:
  edl:
    loss_type: digamma
  mc_dropout:
    num_estimators: 10
    ood_criterion: mi

method_dataset_args:
  edl:
    seu:
      reg_weight: 0.01
      weight_decay: 0.005
    pu:
      reg_weight: 0.001
      weight_decay: 0.001

  mc_dropout:
    seu:
      dropout_rate: 0.1
    pu:
      dropout_rate: 0.2
```

通过 `run.py` 批量运行时，方法参数优先级为：

```text
方法文件默认值 < method_args < method_dataset_args
```

`method_dataset_args` 只能填写方法独有参数，不能覆盖 `epochs`、`lr`、
`batch_size`、`seeds`、`val_split`、硬件设置或评估开关。公共参数继续在
`training`、`hardware` 和 `evaluation` 中统一设置，以保证不同方法和数据集的
实验条件一致。仓库中的 `configs/tuning.yaml` 已保存正式实验的调参结果，
请直接使用，运行前更新其中的 `datasets[].root` 为当前机器的数据路径：

```powershell
python run.py --config configs/tuning.yaml --dry-run
python run.py --config configs/tuning.yaml
```

需要开展新的调参实验时，请将配置复制到其他名称，例如
`configs/my_tuning.yaml`，以保留现有 `tuning.yaml`。

## 参数优先级

运行单个方法时，参数优先级为：

```text
命令行参数 > --config 指定的 YAML > 代码默认值
```

常用公共参数：

| 参数 | 说明 |
|---|---|
| `--config` | YAML 配置路径 |
| `--dataset` | 数据集名称 |
| `--data-root` | 数据集根目录 |
| `--backbone` | 模型名称 |
| `--epochs` | 训练轮数 |
| `--batch-size` | batch size |
| `--lr` | 学习率 |
| `--seeds` | 明确指定随机种子列表 |
| `--n-runs` | `seeds` 未设置时生成 `0..n_runs-1` |
| `--val-split` | 验证集比例 |
| `--eval-noise` | 启用噪声评估 |
| `--no-eval-noise` | 禁用噪声评估 |
| `--accelerator` | `auto`、`gpu` 或 `cpu` |
| `--devices` | 设备数量或 `auto` |
| `--strategy` | 多卡策略 |
| `--precision` | 训练精度 |
| `--output-dir` | 输出根目录 |

## 支持的方法

| 类别 | 方法标识 |
|---|---|
| 基线 | `max_softmax` |
| 集成方法 | `deep_ensemble`, `packed_ensemble`, `batch_ensemble`, `snapshot_ensemble`, `checkpoint_ensemble` |
| 贝叶斯方法 | `variational_bnn`, `swag`, `sgld`, `sghmc` |
| 证据方法 | `edl` |
| 后处理/采样 | `temperature_scaling`, `laplace_approx`, `mc_dropout`, `mc_batch_norm` |

方法名称与 `methods/<方法名称>.py` 一一对应。

EDL 通过 Softplus 将 backbone 输出转换为非负 evidence。训练使用 `DECLoss`，
验证和测试使用 Dirichlet 均值 `alpha / alpha.sum()` 计算 ACC、NLL 和 ECE，
其中 `alpha = evidence + 1`；OOD 检测使用 Dirichlet vacuity `K / alpha.sum()`。

主 OOD 指标使用各方法的原生不确定性准则：确定性基线使用 MSP，EDL 使用
Dirichlet vacuity `K/S`，采样和集成方法使用互信息（MI），Temperature Scaling
和 Laplace 使用后处理概率上的 MSP。

## 支持的模型

公共 CLI 当前接受：

```text
resnet
lenet
mlp
transformer
lstm
timesnet
```

部分方法需要模型提供专用的 Bayesian 或 Packed 变体，因此不保证上述每个模型都能与所有方法组合。

## 支持的数据集

```text
seu
wt
pu
xjtu
hit
cwru
thu
mgb
```

数据输入统一为一维时域信号。一键运行多个数据集时，在 `datasets` 列表中为每个数据集同时填写 `name` 和 `root`。直接运行单个 method 时，默认使用列表中的第一个数据集，也可以通过 `--dataset` 和 `--data-root` 覆盖。

### 数据缓存

所有数据集共用持久化信号缓存。首次运行时，loader 会读取原始
CSV、MAT、TXT 或 NPY 文件并完成确定性的文件级预处理；后续方法、
模型和随机种子会直接复用缓存，不再反复解析原始文件。窗口划分、
数据划分、数据增强和归一化仍按当前实验配置执行，不会被缓存固化。

默认缓存目录为：

~~~text
~/.cache/tfd_bench/datasets/<dataset>/
~~~

源文件路径、大小、修改时间或预处理参数变化时，缓存会自动失效。
如需把缓存放到其他磁盘，可设置环境变量 TFD_DATA_CACHE。


## 输出结构

默认结果保存在：

```text
results/<dataset>/<backbone>/<method>/
├── manifest.json
├── runs.csv
├── summary.csv
└── seed<seed>/
    ├── metrics.csv
    ├── predictions/
    │   ├── clean.npz
    │   └── <noise>_s<severity>.npz
    ├── ckpt/
    └── logs/
```

- `metrics.csv`：单个随机种子的 clean/noise 指标
- `runs.csv`：所有随机种子的原始指标，是方法级结果的统一入口
- `summary.csv`：长表格式的均值、标准差与有效运行数
- `manifest.json`：结果格式版本、运行状态、方法参数与文件索引
- `predictions/*.npz`：逐样本概率、标签及方法原生 OOD 分数，供数值核验使用

`seed<seed>/logs/` 仅保存 Lightning 训练过程日志，不作为结果汇总输入。旧版本的
`raw_all_seeds.csv` 仍可读取；使用当前代码重跑后会自动采用上述统一格式。

启用噪声评估后，每个随机种子还会运行数据集定义的不同噪声类型和严重程度，因此运行时间会明显增加。调试代码时建议使用：

```bash
python methods/max_softmax.py \
  --epochs 1 \
  --seeds 0 \
  --no-eval-noise
```

## 结果分析

汇总实验结果：

```bash
python analysis/collect_results.py
```

默认同时汇总 clean 和所有噪声等级。只读取 clean 或指定噪声等级：

```bash
python analysis/collect_results.py --test-config clean
python analysis/collect_results.py --test-config gaussian_s3
```

也可以筛选数据集、模型或方法：

```bash
python analysis/collect_results.py \
  --dataset seu \
  --backbone resnet \
  --method edl
```

汇总保存到 `results/summary.json`。结果收集包含 MSP 和论文中的 13 种不确定性方法，
表格默认展示 TS、VBNN、SGLD、SGHMC、DE、SE、BE、PE、SWAG、LA、MCD、MCBN 和 EDL。
Checkpoint Ensemble 的原始实验结果仍保留，但不进入默认汇总和表格。
训练命令不会自动执行结果汇总。ECE 评估使用 10 个分箱。

也可以单独生成对比表格：

```bash
python analysis/generate_tables.py
```

Markdown 表格默认保存到 `results/tables/table.md`。表格只展示 `ACC`、`ECE`、`AUROC`，并按数据集、模型和测试配置分别生成表格，例如 `clean`、`gaussian_s1`、...、`gaussian_s5` 各有一张表。ACC、ECE 和 AUROC 均以百分数显示，粗体表示同一张表中的最优结果。

OOD 评估同时保存总体 AUROC、各 OOD 故障/文件来源的 AUROC，以及对来源等权平均的
Macro AUROC。需要生成包含这些明细的表格时，可显式指定指标，例如：

```bash
python analysis/generate_tables.py --metrics \
  test/cls/Acc test/cal/ECE \
  ood/overall_AUROC ood/source_AUROC/ORS1_var \
  ood/source_AUROC/IRS1_var ood/source_AUROC/CC1_var \
  ood/macro_AUROC
```

同时在终端中显示表格，或生成其他格式：

```bash
python analysis/generate_tables.py --print
python analysis/generate_tables.py --format latex
python analysis/generate_tables.py --format html
```

后两条命令默认分别保存到 `results/tables/table.tex` 和 `results/tables/table.html`；仍可使用 `--output` 指定其他保存位置。

分步生成命令等价于：

```bash
python analysis/collect_results.py
python analysis/generate_tables.py
```

常规方法使用验证集 `NLL` 最小的 epoch 作为最佳 checkpoint。SGLD、SGHMC 和
SWAG 属于 posterior sampling 方法，评估训练结束时形成的完整样本集合，不加载某个
单独 epoch 的最佳 checkpoint；它们的预训练模型仍由验证集 NLL 选择。测试集以及
噪声测试结果均不参与模型选择，多个随机种子独立运行后再汇总均值和标准差。


## 数据准备与复现

数据集不随仓库分发，代码的 Apache-2.0 许可证也不覆盖第三方数据。请先阅读：

- [`docs/DATASETS.md`](docs/DATASETS.md)：八个 loader 所需目录、文件名和已核实的官方来源
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)：环境记录、smoke test、完整实验、结果归档和复现边界

运行前请将默认或调参配置中的 `datasets[].root` 更新为当前机器的数据路径。
`data/` 已加入 `.gitignore`。

无数据即可执行的发布前检查：

```bash
python -m compileall -q run.py methods src analysis tests
python -m unittest discover -s tests -v
python run.py --dry-run
```

GitHub Actions 会在 push 和 pull request 时执行相同检查，并额外验证全部 method 的 `--help` 入口。

## 开源许可证与致谢

TFD-Bench 以 [Apache License 2.0](LICENSE) 发布，第三方归属见 [`NOTICE`](NOTICE)。

本项目的部分不确定性量化训练、模型包装、指标和后处理组件基于或改写自
[TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty)，并针对一维故障诊断 benchmark 做了裁剪和修改。TorchUncertainty 同样采用 Apache-2.0 许可证。

如果这些组件对研究有帮助，请引用 TorchUncertainty：

```bibtex
@inproceedings{lafage2025torch_uncertainty,
  title     = {Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification},
  author    = {Lafage, Adrien and Laurent, Olivier and Gabetni, Firas and Franchi, Gianni},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2025}
}
```

各故障数据集仍应分别引用其原始发布方；本仓库不授予任何第三方数据的再分发权。

# Dataset Preparation

TFD-Bench does not redistribute datasets. Users must obtain every dataset from its original provider and comply with the provider's license, citation and access terms. The repository's Apache-2.0 license applies to the code, not to third-party datasets.

Set `datasets[].root` in the YAML to the directory described below. Paths may be absolute or relative to the repository root.

## Dataset summary

| ID | Loader input | Expected root |
|---|---|---|
| `seu` | Flat CSV recordings | Directory containing the listed bearing and gear CSV files |
| `mgb` | Flat CSV recordings | Directory containing the listed MGB CSV files |
| `wt` | MATLAB files grouped by fault folder | Directory containing `broken/`, `healthy/`, `missing_tooth/`, `root_crack/`, `wear/` |
| `pu` | Paderborn MATLAB hierarchy | The extracted `PU-dataset-main/` directory |
| `xjtu` | Class folders containing two text channels | Directory containing the nine `1ndBearing_*`/`2ndPlanetary_*` folders |
| `hit` | Five NumPy arrays | Directory containing `data1.npy` through `data5.npy` |
| `cwru` | CWRU 12 kHz drive-end MATLAB files | Extracted `12k/` directory containing numbered `.mat` files |
| `thu` | Flat CSV recordings | Directory containing files formed from the configured fault prefixes and operating-condition suffixes |

## SEU

## Leakage-resistant split protocol

Dataset partitioning is deterministic and independent of the model-training
seed. SEU, MGB, PU, CWRU, WT, XJTU and HIT are split within each source
recording in chronological order. The early segment is used for training, the
following segment for validation and the final segment for testing. One complete
window is discarded at each split boundary so adjacent windows cannot occur in
different subsets. Splitting is performed before recordings are concatenated,
which preserves the class and source composition of every subset.

THU provides separate files for training, validation and testing. Its loader
therefore uses file-level isolation and raises an error if a source filename is
assigned to more than one subset.

Every temporal split retains source and window-index metadata for automatic
overlap auditing. Repeated DataModule construction produces the same partition
regardless of the uncertainty method, backbone or training seed.


Expected files are flat under the configured root. ID files include:

```text
ball_20_0.csv
comb_20_0.csv
health_20_0.csv
inner_20_0.csv
outer_20_0.csv
```

OOD files are `Chipped`, `Miss`, `Root` and `Surface` recordings at conditions `20_0` and `30_2`. Shift files use the ID fault names at condition `30_2`. The loader skips the first 16 lines and reads the second signal column; preserve the original delimiters.

## MGB

Expected files are flat CSV recordings. The current benchmark uses six ID files at `2500_0`, selected shift files at `3000_40`, and these OOD sources:

```text
ORS1_var.csv
IRS1_var.csv
CC1_var.csv
```

The loader reads the second CSV column and decimates the signal by four. Do not pre-decimate the files or the benchmark preprocessing will be applied twice.

## WT

Expected hierarchy:

```text
<root>/
├── broken/B1_30.MAT
├── healthy/N1_30.MAT
├── missing_tooth/M1_30.MAT
├── root_crack/R1_30.MAT
└── wear/W1_30.MAT
```

The benchmark uses only Assembly 1. ID recordings contain `broken`, `healthy`
and `missing_tooth` at 30, 35 and 40 Hz. Operating shift retains the same
fault categories and assembly but uses 20, 25, 50 and 55 Hz. OOD recordings
contain `root_crack` and `wear` at the three ID speeds. This isolates speed
shift from changes caused by a different assembly.

The default WT limits are 500 ID windows, 100 OOD windows and 100
operating-shift windows per raw file. Windows are selected uniformly over the
complete recording before the chronological train/validation/test split. These
limits can be changed with the dataset-level YAML keys
`max_id_windows_per_file`, `max_ood_windows_per_file` and
`max_shift_windows_per_file`. Every MATLAB file must contain a `Data` array
with four channels.

## PU

Official source: [Paderborn University Bearing Data Center](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter).

Point the root to the extracted directory containing operating-condition folders such as:

```text
PU-dataset-main/
├── N09_M07_F10/
├── N15_M01_F10/
├── N15_M07_F04/
└── N15_M07_F10/
```

Keep the original `.mat` names, for example `N15_M07_F10_KA04_1.mat`.

## XJTU

Expected hierarchy:

```text
<root>/
├── 1ndBearing_ball/
├── 1ndBearing_inner/
├── 1ndBearing_mix(inner+outer+ball)/
├── 1ndBearing_outer/
├── 2ndPlanetary_brokentooth/
├── 2ndPlanetary_missingtooth/
├── 2ndPlanetary_normalstate/
├── 2ndPlanetary_rootcracks/
└── 2ndPlanetary_toothwear/
```

Each class folder must contain `Data_Chan1.txt` and `Data_Chan2.txt`. Channel 1 is used for ID/OOD evaluation and channel 2 for distribution shift. The first 14 header lines are skipped.

The default XJTU limits are 500 ID windows per class file, 800 OOD windows from
the compound-fault file and 500 shift windows per class file. Uniform selection
is applied over the complete recording before temporal splitting. The limits
use the same max_id_windows_per_file, max_ood_windows_per_file and
max_shift_windows_per_file YAML keys as WT.


## HIT

Expected flat files:

```text
data1.npy  # ID normal
data2.npy  # shifted normal
data3.npy  # ID inner fault
data4.npy  # shifted inner fault
data5.npy  # OOD outer fault
```

Each array is expected to have shape `N × 6 × 20480`. The loader selects channels and truncates signals to the configured one-dimensional window length.

## CWRU

Official source: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter).

Point the root to the extracted 12 kHz drive-end directory containing numbered files such as `97.mat`, `105.mat`, `118.mat` and `130.mat`. Keep the original MATLAB variable names; the loader resolves the drive-end signal key from the file number.

## THU

Expected files are flat CSV recordings. Names are constructed as:

```text
<fault-prefix><condition-suffix>
```

ID prefixes include `health_`, `gear_pitting_H_`, `gear_wear_H_`, `teeth_break_H_` and `teeth_crack_H_`. OOD prefixes are the two configured compound gear/bearing faults. Required suffixes are defined in `src/datasets/thu.py`; the default split uses separate speed-circulation and torque-circulation recordings.

## Dataset integrity checklist

Before a full run:

1. Confirm every required source file exists under the configured root.
2. Keep the original sample order, units, delimiter and MATLAB keys.
3. Do not place generated `.npy` caches or transformed copies in Git.
4. Run one seed of `max_softmax` with `--epochs 1 --no-eval-noise`.
5. Confirm that `manifest.json`, `runs.csv` and `seed0/predictions/clean.npz` are produced.
6. Record the dataset provider, version/download date and citation with published results.

MGB, WT, XJTU, HIT, SEU and THU source distributions are not mirrored here. If their provider requires an application, institutional access or a separate agreement, those terms take precedence over this documentation.
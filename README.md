# IDS-ReComm: dataset and model evaluation for intrusion detection

Code and results for:

> I. Sanober and R. N. Mir, "A Comprehensive Evaluation of IDS Datasets and Models
> with a Derived Framework for Informed Dataset Selection: IDS-ReComm,"
> *Advances in Computing and Engineering*. Under review.

The study evaluates five classical machine learning models and five deep learning
models across four benchmark IDS datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017,
HIKARI-2021) plus one ablation variant, under six configurations per dataset,
repeated over five random seeds. Total: **1,500 runs**.

---

## What this repository contains

```
ids_recomm/          pipeline: loading, preprocessing, models, metrics, analysis
slurm/               batch submission scripts (SLURM)
results/             per-run metrics, tidy and summary CSVs, statistical output
figures/             every figure in the paper, and the script that draws them
docs/                reproduction notes and a description of the audit checks
```

Per-run predicted labels and probabilities for all 1,500 runs are ~0.6 GB and
are archived separately on Zenodo (DOI below), not in this repository.

---

## The pipeline in one paragraph

Every transform is fitted on training data alone. Cleaning and deduplication are
followed by a stratified 70/30 train-test split, then a stratified 90/10
train-validation split taken **before** any resampling. The one-hot encoder,
correlation filter, recursive feature elimination and standardiser are each fitted
on the training rows and applied outward to validation and test data by
transformation. SMOTE touches the training split only. No step downstream of the
split observes a test row.

This ordering matters. Fitting feature selection before the split leaks test labels
into the feature choice, and the effect is not small: see `docs/AUDIT.md`.

---

## Reproducing the results

### 1. Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export IDS_RAW=/path/to/datasets      # where the raw CSVs live
export IDS_WORK=/path/to/scratch      # cache, results and predictions
```

Tested with Python 3.11. TensorFlow is pinned to a CPU build: all reported timings
are CPU-only, with GPU acceleration explicitly disabled, so that classical and deep
models are compared on identical hardware.

### 2. Data

The four datasets are public but not redistributed here. Download them and set
`RAW_DIR` in `ids_recomm/config.py`:

| Dataset | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| HIKARI-2021 | https://zenodo.org/record/6463389 |

For CIC-IDS2017 use the `MachineLearningCVE` distribution (79 columns), combined
into a single CSV.

### 3. Preflight

```bash
python ids_recomm/check_config.py
```

Do not skip this. It verifies every path, confirms the label column and its benign
values, and flags columns whose names suggest leakage. It also reports label churn:
a value near zero means the file is ordered by class, in which case any sequential
index column is a near-perfect predictor of the label. HIKARI-2021 is such a file.

All five datasets must report PASS before proceeding.

### 4. Run

```bash
# stage 1: clean raw CSVs into parquet          (5 tasks)
sbatch slurm/01_prepare.sbatch

# stage 2: fit transforms per dataset and seed  (25 tasks)
sbatch --dependency=afterok:<JOBID> slurm/02_prep.sbatch

# stage 3: the grid                             (50 group tasks, 1,500 runs)
python ids_recomm/make_manifest.py
sbatch --array=1-50 slurm/03_run.sbatch
```

Without a scheduler, run the group tasks in a loop:

```bash
while read -r args; do python ids_recomm/run_group.py $args; done < manifest.txt
```

Both stages skip completed work, so an interrupted run resumes where it stopped.
`make_manifest.py --skip-done` rebuilds a manifest of only what is missing.

Expected wall-clock: roughly 20 to 25 hours in total, dominated by CIC-IDS2017.
The published results were produced on a single cluster node providing 64 physical
CPU cores on one socket and 503 GB of RAM, with 16 cores and 128 GB allocated per
job and GPU acceleration disabled.

### 5. Results and figures

```bash
python ids_recomm/aggregate.py            # tidy CSV, summary CSV, audit report
python ids_recomm/analyze.py --metric f1  # also: pr_auc, accuracy, roc_auc,
                                          # train_seconds, predict_seconds
python ids_recomm/make_figures.py --input results/results_tidy.csv --outdir figures
```

---

## The audit

`aggregate.py` writes `results/audit_report.txt`. Five checks run on every
aggregation, and all pass on the released result set:

| Check | What it catches |
|---|---|
| F1 equals the harmonic mean of precision and recall | transposed metric columns |
| No two reported metrics are identical across the set | a metric reported twice under different names |
| Both model families satisfy the same metric identities | families evaluated by different definitions |
| Every configuration has all five seeds | partial runs reported as complete |
| ROC-AUC range per family | ceiling effects that make significance tests uninformative |

These are not decoration. The first two exist because the earlier version of this
work contained exactly those two errors; `docs/AUDIT.md` describes them.

---

## Headline findings

- Dataset choice explains **9.5%** of the variance in F1-score for classical models
  and **94.3%** for deep models (eta-squared, repeated-measures ANOVA).
- **Random Forest** attains the highest F1-score on all four datasets, and among the
  lowest inference cost.
- **HIKARI-2021 ranks first on accuracy and last on F1-score.** Its effective class
  imbalance after correct deduplication is **34.0:1**, not the 13.7:1 the distributed
  file implies, because 62.7% of its attack flows are exact duplicates. That
  duplication is undetectable while the per-flow `uid` column is retained.
- Removing network identifiers from HIKARI-2021, on an identical row set, changes
  F1-score by **under one percentage point for eight of ten models**.

---

## A note on HIKARI-2021

Three properties of the distributed file affect any evaluation using it:

1. It carries two row-index columns (`Unnamed: 0`, `Unnamed: 0.1`) and a per-flow
   identifier (`uid`). The file is ordered by class, so a row index predicts the
   label almost perfectly. All three are dropped here.
2. Duplicate detection performed while `uid` is present cannot fire, since `uid` is
   unique per row. Deduplication must run on the feature set.
3. Of 64,030 feature-identical flows, 34,246 carry contradictory labels. This
   pipeline retains the first occurrence, which is order-dependent given the class
   ordering; `CONTRADICTION_POLICY` in `config.py` switches to removing all members
   of ambiguous groups, which leaves 3,290 attack flows at 137.9:1.

---

## Data archive

Per-run predicted labels and probabilities for all 1,500 runs, and the cleaned
parquet files, are archived at:

**DOI: [ ZENODO DOI ]**

Download and unpack into `predictions/` to recompute any metric without retraining:

```python
import numpy as np
z = np.load("predictions/CIC-IDS2017__balanced__all__RF__seed42.npz")
z["y_true"], z["y_pred"], z["y_prob"]
```

---

## Citation

See `CITATION.cff`. If you use the pipeline, please cite the paper; if you use the
archived predictions, please also cite the Zenodo record.

## License

Code is released under the MIT License (`LICENSE`). The datasets analysed remain
under the licences of their original providers and are not redistributed here.

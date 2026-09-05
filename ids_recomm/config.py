"""
Central configuration for the IDS-ReComm corrected re-run.

EDIT THE PATHS IN `DATASETS` BEFORE RUNNING ANYTHING ELSE.
Everything else has sane defaults matching the original study's protocol.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Root for all generated artefacts. Point this at HPC scratch, not $HOME.
# Set both via environment variables, or edit them here.
#   export IDS_WORK=/scratch/$USER/ids_rerun
#   export IDS_RAW=/scratch/$USER/ids_datasets
WORK = Path(os.environ.get("IDS_WORK", "./work")).expanduser()

RAW_DIR   = Path(os.environ.get("IDS_RAW", "./raw")).expanduser()
CACHE_DIR = WORK / "cache"      # cleaned parquet + per-seed prep artefacts
RESULT_DIR = WORK / "results"   # one JSON per (dataset, seed, features, balance, model)
PRED_DIR   = WORK / "predictions"  # y_true / y_pred / y_prob npz per run
LOG_DIR    = WORK / "logs"

for _d in (CACHE_DIR, RESULT_DIR, PRED_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]        # 5 repeated runs (Reviewer C4)
TEST_SIZE = 0.30                     # unchanged from original study
VAL_SIZE_OF_TRAIN = 0.10             # carved out of the 70% train, BEFORE SMOTE
# Categorical columns with more than this many distinct values get ordinal
# encoding instead of one-hot. The threshold exists solely to stop HIKARI-2021's
# originh (2,899 distinct) and responh (7,991) from producing ~10,900 dense
# columns. It must stay well above the legitimate categoricals: NSL-KDD's
# `service` has 70 distinct values and UNSW-NB15's `proto` has 133, and both
# were one-hot encoded in the original notebooks.
CARDINALITY_MAX = 200

# How to resolve rows whose FEATURES are identical but whose LABELS disagree.
#   "drop_all"   - remove every copy. A flow that appears with both labels is
#                  unresolvable, so it teaches nothing and is removed. This is
#                  order-independent and matches the intent of the original
#                  notebooks' drop_duplicates(keep=False).
#   "keep_first"  - keep whichever copy appears first in the file. Cheaper, but
#                  ORDER-DEPENDENT: HIKARI-2021 is class-ordered with attacks
#                  first, so every contradictory pair resolves in favour of the
#                  attack label and the test-set class balance becomes an
#                  artifact of file ordering.
# Consistent duplicates (same features AND same label) are always collapsed to
# one copy under both settings.
CONTRADICTION_POLICY = "drop_all"
CORR_THRESHOLD = 0.95                # unchanged; now computed on TRAIN ONLY, label excluded

FEATURE_SETTINGS = ["all", "top20", "top10"]
BALANCE_SETTINGS = ["imbalanced", "balanced"]

# DL training: uniform protocol across all datasets (answers the "unequal epoch
# allocation" limitation in section 5.5 and Reviewer C's tuning comment).
DL_MAX_EPOCHS = 200
DL_BATCH_SIZE = 2000
DL_PATIENCE = 10                     # EarlyStopping on real (non-SMOTE) val_loss
DL_MIN_DELTA = 1e-4                  # improvements below this do not reset patience

# WATCH `epochs_run` IN THE RESULTS. The original study used 400 epochs for
# NSL-KDD/UNSW-NB15 and 8-10 for CIC-IDS2017/HIKARI-2021, which Reviewer C
# flagged. EarlyStopping replaces that with one adaptive rule, but if a model
# runs to DL_MAX_EPOCHS on a 2M-row dataset the cost is large. If the smoke
# test shows CIC-IDS2017 hitting the cap, revisit before the full grid.

# ---------------------------------------------------------------------------
# >>> THE ONE SUBSTANTIVE PROTOCOL CHANGE - REQUIRES YOUR SIGN-OFF <<<
# RBF-kernel SVC is O(n^2)-O(n^3). In the original run SVM on CIC-IDS2017 alone
# consumed 86.8 of the 98.4 total compute hours (88%). Capping the SVM training
# set at a stratified 100k sample, applied UNIFORMLY to all four datasets, keeps
# the protocol consistent across datasets and reduces total wall-clock by ~85%.
# Set to None to disable the cap and reproduce the original (4+ day) behaviour.
SVM_MAX_TRAIN = 100_000
# ---------------------------------------------------------------------------

ML_MODELS = ["DT", "RF", "GNB", "KNN", "SVM"]
DL_MODELS = ["GRU", "LSTM", "MLP", "SimpleRNN", "CNN"]
ALL_MODELS = ML_MODELS + DL_MODELS

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
# `label_col`      : raw column holding the class
# `benign_values`  : values in label_col that mean "not an attack" (lowercased)
# `drop_cols`      : leakage / bookkeeping columns removed before any modelling
# `identifier_cols`: network identifiers, dropped only in the *_noid ablation
DATASETS = {
    # File names below match the layout used in this study. Adjust `files` to
    # your own copies; `label_col`, `benign_values` and `drop_cols` should not
    # be changed, as they were verified against the distributed data with
    # check_config.py.
    "NSL-KDD": dict(
        derive_from=None,
        files=[RAW_DIR / "NSL-KDD" / "NSL-KDD1.csv"],
        read_kwargs=dict(),
        column_names=None,              # the file carries its own header row
        label_col="attack_map",
        benign_values={"normal"},
        # 'attack' is a direct restatement of 'attack_map'; 'level' is the
        # difficulty score and is not a traffic feature
        drop_cols=["attack", "level"],
        identifier_cols=[],
    ),
    "UNSW-NB15": dict(
        derive_from=None,
        files=[RAW_DIR / "UNSW-NB15" / "UNSW_NB15_training-set.csv",
               RAW_DIR / "UNSW-NB15" / "UNSW_NB15_testing-set.csv"],
        read_kwargs=dict(),
        column_names=None,
        label_col="label",
        benign_values={"0", "0.0"},
        # 'attack_cat' restates the label; 'id' is a per-row index and, because
        # the file is largely class-ordered, predicts the label
        drop_cols=["id", "attack_cat"],
        identifier_cols=[],
    ),
    "CIC-IDS2017": dict(
        derive_from=None,
        # the MachineLearningCVE distribution (79 columns), concatenated
        files=[RAW_DIR / "CICIDS2017" / "combined_file.csv"],
        read_kwargs=dict(encoding="latin-1", low_memory=False),
        column_names=None,
        # NOTE: the raw header is " Label" with a leading space. Column names
        # are stripped on load, so "Label" is correct here.
        label_col="Label",
        benign_values={"benign"},
        # this distribution carries no flow, address, port or timestamp fields;
        # 'Fwd Header Length.1' duplicates 'Fwd Header Length' and is removed
        # automatically by the correlation filter
        drop_cols=[],
        identifier_cols=[],
    ),
    "HIKARI-2021": dict(
        derive_from=None,
        files=[RAW_DIR / "HIKARI" / "ALLFLOWMETER_HIKARI2021.csv"],
        read_kwargs=dict(),
        column_names=None,
        label_col="Label",
        benign_values={"0", "0.0"},
        # 'traffic_category' restates the label. 'Unnamed: 0' and
        # 'Unnamed: 0.1' are row indices and the file is ordered by class, so
        # either predicts the label almost perfectly. 'uid' is unique per flow
        # (555,278 distinct values), which also defeats duplicate detection.
        drop_cols=["Unnamed: 0.1", "Unnamed: 0", "uid", "traffic_category"],
        identifier_cols=["originh", "originp", "responh", "responp"],
    ),
}

DATASETS["HIKARI-2021-noID"] = dict(
    DATASETS["HIKARI-2021"],
    derive_from="HIKARI-2021",
    drop_cols=DATASETS["HIKARI-2021"]["drop_cols"]
    + DATASETS["HIKARI-2021"]["identifier_cols"],
)

DATASET_NAMES = list(DATASETS.keys())

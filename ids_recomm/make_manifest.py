"""
Build the task manifest consumed by the SLURM array job.

    python make_manifest.py                 # full grid
    python make_manifest.py --datasets NSL-KDD --seeds 42   # smoke test

Emits one task per (dataset, seed, balance) group - 50 tasks, each covering
3 feature settings x 10 models. Grouping matters because the cluster allows
only one concurrent job per user, so tasks run serially and per-task
preprocessing overhead would otherwise dominate.

Emitted longest-expected-first so the expensive CIC-IDS2017 groups start early
and any queue interruption costs the cheap tail rather than the expensive head.
"""
import argparse
import itertools
from pathlib import Path

import config as C
from run_experiment import tag_for

# rough relative cost, taken from the original study's reported timings
DATASET_COST = {"CIC-IDS2017": 100, "HIKARI-2021": 6, "HIKARI-2021-noID": 6,
                "UNSW-NB15": 4, "NSL-KDD": 1}
MODEL_COST = {"SVM": 100, "KNN": 30, "CNN": 12, "RF": 8, "GRU": 6, "LSTM": 6,
              "SimpleRNN": 4, "MLP": 3, "DT": 2, "GNB": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=C.DATASET_NAMES)
    ap.add_argument("--seeds", nargs="*", type=int, default=C.SEEDS)
    ap.add_argument("--models", nargs="*", default=C.ALL_MODELS)
    ap.add_argument("--out", default="manifest.txt")
    ap.add_argument("--skip-done", action="store_true",
                    help="omit configurations whose result JSON already exists")
    a = ap.parse_args()

    rows = []
    for ds, seed, bal in itertools.product(a.datasets, a.seeds, C.BALANCE_SETTINGS):
        if a.skip_done:
            pending = any(
                not (C.RESULT_DIR / f"{tag_for(ds, seed, f, bal, m)}.json").exists()
                for f in C.FEATURE_SETTINGS for m in a.models)
            if not pending:
                continue
        cost = DATASET_COST.get(ds, 1) * sum(MODEL_COST.get(m, 1) for m in a.models)
        rows.append((cost, ds, seed, bal))

    rows.sort(key=lambda r: -r[0])
    lines = [f"--dataset {ds} --seed {s} --balance {b}" for _, ds, s, b in rows]
    Path(a.out).write_text("\n".join(lines) + "\n")

    print(f"{len(lines)} group tasks written to {a.out}")
    print(f"  (each covers {len(C.FEATURE_SETTINGS)} feature settings x "
          f"{len(a.models)} models = {len(C.FEATURE_SETTINGS)*len(a.models)} runs)")
    print(f"submit with:  sbatch --array=1-{len(lines)} slurm/03_run.sbatch")


if __name__ == "__main__":
    main()

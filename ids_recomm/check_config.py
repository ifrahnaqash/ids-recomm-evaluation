"""
Preflight. Run this BEFORE submitting anything to the scheduler.

    python check_config.py

Verifies that every raw file exists, that the configured label column is
present, that the benign values actually match something, and prints the column
headers so the drop lists in config.py can be checked against reality rather
than against my guess at your CSV layout.

It also flags columns whose name pattern suggests leakage. Every dataset except
CIC-IDS2017 in your earlier IDS-ReComm work carried at least one leakage column
producing perfect scores, so this is worth two minutes.
"""
import sys

import pandas as pd

import config as C

import re

# Full-name patterns, not loose substrings. The earlier substring version
# matched "id" inside "idle.min" and "port" inside "dst_host_same_src_port_rate",
# which buried the real findings under false positives.
SUSPECT = [
    (r"^unnamed", "index column written by pandas"),
    (r"^(uid|id|index|row_?id|rowid)$", "row identifier"),
    (r"^flow[ _]?id$", "flow identifier"),
    (r"^(source|src|destination|dst)[ _]?(ip|port)$", "network identifier"),
    (r"^(originh|originp|responh|responp)$", "network identifier"),
    (r"^(timestamp|date|time)$", "temporal identifier; leaks capture ordering"),
    (r"^(label|class|category|attack|attack_cat|traffic_category)$",
     "restates the target"),
    (r"_map$", "may restate the target"),
]


def label_profile(files, label_col, read_kwargs, column_names):
    """Read ONLY the label column, across every file, in full.

    Resolves the column by POSITION, not by name. `usecols` matches the raw
    header, and several IDS CSVs ship with leading spaces in their headers
    (CIC-IDS2017's label column is literally " Label"). Everything else in this
    codebase compares against stripped names, so the lookup is done here by
    stripping the header and taking the index of the match.

    Reading 2000 rows of a file sorted by label reports an attack rate of 0.000
    or 1.000 and makes benign_values look broken when it is fine. One column is
    cheap to read in full even at 2.8M rows.
    """
    parts = []
    for f in files:
        kw = dict(read_kwargs)
        if column_names:
            header = list(column_names)
            kw["names"] = column_names
        else:
            probe = dict(read_kwargs)
            probe.pop("low_memory", None)
            header = [str(c).strip()
                      for c in pd.read_csv(f, nrows=0, **probe).columns]
        if label_col not in header:
            raise KeyError(f"{label_col!r} not in header of {f}")
        kw["usecols"] = [header.index(label_col)]
        kw.pop("low_memory", None)
        s = pd.read_csv(f, **kw).iloc[:, 0]
        parts.append(s)
    return pd.concat(parts, ignore_index=True).astype(str).str.strip().str.lower()


def sortedness(y: pd.Series) -> float:
    """Fraction of adjacent rows whose label differs.

    Near zero means the file is ordered by class. That matters: in a sorted
    file any sequential index column is a near-perfect predictor of the label,
    so an 'Unnamed: 0' left in the feature set is total leakage.
    """
    if len(y) < 2:
        return 1.0
    return float((y.values[1:] != y.values[:-1]).mean())


def check(name: str) -> bool:
    spec = C.DATASETS[name]
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    ok = True

    files = list(spec["files"])
    if not files:
        print("  FAIL  no files configured (glob matched nothing?)")
        return False

    for f in files:
        print(f"  {'OK  ' if f.exists() else 'FAIL'}  {f}")
        ok &= f.exists()
    if not ok:
        return False

    kw = dict(spec["read_kwargs"])
    if spec["column_names"]:
        kw["names"] = spec["column_names"]
    head = pd.read_csv(files[0], nrows=2000, **kw)
    head.columns = [str(c).strip() for c in head.columns]

    print(f"\n  {len(head.columns)} columns:")
    for i in range(0, len(head.columns), 4):
        print("     " + "".join(f"{c:<28}" for c in head.columns[i:i + 4]))

    label = spec["label_col"]
    if label not in head.columns:
        print(f"\n  FAIL  label column {label!r} not in headers")
        return False
    print(f"\n  OK    label column: {label!r}")

    y = label_profile(files, label, spec["read_kwargs"], spec["column_names"])
    vals = y.value_counts()
    benign = {b.lower() for b in spec["benign_values"]}
    print(f"  label values (full column, {len(y):,} rows): {dict(list(vals.items())[:10])}")
    matched = [v for v in vals.index if v in benign]
    if not matched:
        print(f"  WARN  none of benign_values={benign} matched any label value.")
        print(f"        Every row would be labelled 'attack'. Fix benign_values.")
        ok = False
    else:
        rate = 1 - vals[matched].sum() / vals.sum()
        print(f"  OK    benign matched {matched}; implied attack rate ~{rate:.3f}")

    churn = sortedness(y)
    print(f"  label churn (adjacent rows differing): {churn:.4f}")
    if churn < 0.01:
        print("  WARN  file appears ORDERED BY CLASS. Any sequential index column")
        print("        (e.g. 'Unnamed: 0') is then a near-perfect predictor of the")
        print("        label. Confirm every index column is in drop_cols.")

    configured = {c.lower() for c in spec["drop_cols"]} | {label.lower()}
    flags = []
    for c in head.columns:
        cl = c.lower()
        if cl in configured:
            continue
        for pat, why in SUSPECT:
            if re.search(pat, cl):
                flags.append((c, why))
                break
    if flags:
        print(f"\n  REVIEW  {len(flags)} column(s) look like leakage but are NOT in drop_cols:")
        for c, why in flags:
            print(f"            {c:<32} {why}")
        print("          Decide deliberately whether each belongs in drop_cols.")
    else:
        print("\n  OK    no unhandled leakage-pattern columns")

    if spec["identifier_cols"]:
        present = [c for c in spec["identifier_cols"] if c in head.columns]
        missing = [c for c in spec["identifier_cols"] if c not in head.columns]
        print(f"  identifier_cols present: {present}")
        if missing:
            print(f"  WARN  identifier_cols not found in headers: {missing}")
            print(f"        The -noID ablation would then be identical to the base "
                  f"dataset, silently producing a null result.")
            ok = False

    return ok


def main():
    print(f"RAW_DIR   = {C.RAW_DIR}")
    print(f"WORK      = {C.WORK}")
    print(f"SEEDS     = {C.SEEDS}")
    print(f"SVM cap   = {C.SVM_MAX_TRAIN}")

    results = {n: check(n) for n in C.DATASET_NAMES}

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    n_runs = (len(C.DATASET_NAMES) * len(C.FEATURE_SETTINGS)
              * len(C.BALANCE_SETTINGS) * len(C.ALL_MODELS) * len(C.SEEDS))
    print(f"\n  full grid = {n_runs} runs")
    if not all(results.values()):
        print("\n  Fix the FAIL items in config.py before submitting.")
        sys.exit(1)
    print("\n  Preflight passed. Next: sbatch slurm/01_prepare.sbatch")


if __name__ == "__main__":
    main()

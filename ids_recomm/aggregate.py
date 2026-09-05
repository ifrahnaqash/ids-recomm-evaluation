# -*- coding: utf-8 -*-
"""
Stage 4 - collect every result JSON into one tidy table, then emit the
manuscript-shaped tables (mean over seeds, with standard deviation).

    python aggregate.py

Outputs:
    results_tidy.csv        one row per (dataset, balance, features, model, seed)
    results_summary.csv     mean +/- sd over seeds
    tables/Table_*.md       manuscript-shaped tables, ready to paste
    audit_report.txt        automated consistency checks

The audit report exists because the original submission's errors were exactly
the kind a machine catches instantly and a human reading proofs does not:
a metric equal to another metric, a transposed column pair, a caption that
does not match its contents.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

METRICS = ["accuracy", "precision", "recall", "f1", "f1_macro",
           "balanced_accuracy", "mcc", "roc_auc", "pr_auc",
           "train_seconds", "predict_seconds"]

PRETTY = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
          "f1": "F1-Score", "f1_macro": "Macro F1", "roc_auc": "ROC-AUC",
          "pr_auc": "PR-AUC", "mcc": "MCC",
          "balanced_accuracy": "Balanced Accuracy",
          "train_seconds": "Training Time (s)",
          "predict_seconds": "Testing Time (s)"}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(C.RESULT_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        r.pop("feature_names", None)
        rows.append(r)
    if not rows:
        raise SystemExit(f"no results found in {C.RESULT_DIR}")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "balance", "features", "model", "family"]
    agg = df.groupby(keys)[METRICS].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def manuscript_tables(s: pd.DataFrame, outdir: Path):
    outdir.mkdir(exist_ok=True)
    order_ml = C.ML_MODELS
    order_dl = C.DL_MODELS
    feat_order = ["all", "top20", "top10"]
    feat_label = {"all": "All", "top20": "Top20", "top10": "Top10"}
    datasets = [d for d in C.DATASET_NAMES if d in set(s.dataset)]

    n = 7
    for metric in ["accuracy", "precision", "recall", "f1", "pr_auc",
                   "roc_auc", "train_seconds", "predict_seconds"]:
        for fam, order in (("Machine Learning", order_ml), ("Deep Learning", order_dl)):
            lines = [f"**Table {n}. {PRETTY[metric]} for {fam} Models across the "
                     f"Datasets (mean +/- sd over {len(C.SEEDS)} seeds)**", ""]
            head = "| Model | Features | " + " | ".join(
                f"{d} Imbal | {d} Bal" for d in datasets) + " |"
            lines += [head, "|" + "---|" * (2 + 2 * len(datasets))]
            for mdl in order:
                for i, feat in enumerate(feat_order):
                    cells = [mdl if i == 0 else "", feat_label[feat]]
                    for d in datasets:
                        for bal in ["imbalanced", "balanced"]:
                            r = s[(s.dataset == d) & (s.balance == bal)
                                  & (s.features == feat) & (s.model == mdl)]
                            if r.empty:
                                cells.append("--")
                            else:
                                mu = r[f"{metric}_mean"].iloc[0]
                                sd = r[f"{metric}_std"].iloc[0]
                                fmt = ("{:.1f} +/- {:.1f}" if "seconds" in metric
                                       else "{:.2f} +/- {:.2f}")
                                scale = 1 if "seconds" in metric else 100
                                cells.append(fmt.format(mu * scale,
                                                        (sd or 0) * scale))
                    lines.append("| " + " | ".join(cells) + " |")
            (outdir / f"Table_{n:02d}_{metric}_{fam.split()[0]}.md").write_text(
                "\n".join(lines) + "\n")
            n += 1
    print(f"wrote manuscript tables to {outdir}/")


def audit(df: pd.DataFrame, s: pd.DataFrame) -> str:
    out = ["AUTOMATED CONSISTENCY AUDIT", "=" * 60, ""]

    # Completeness is judged against the datasets and seeds actually PRESENT,
    # not against the full configured grid. Otherwise a one-dataset smoke test
    # always reports hundreds of "missing" runs it was never asked to produce,
    # which buries the checks that matter.
    import itertools
    ds_seen = sorted(set(df.dataset))
    seeds_seen = sorted(set(df.seed))
    exp = (len(ds_seen) * len(C.FEATURE_SETTINGS) * len(C.BALANCE_SETTINGS)
           * len(C.ALL_MODELS) * len(seeds_seen))
    full = (len(C.DATASET_NAMES) * len(C.FEATURE_SETTINGS)
            * len(C.BALANCE_SETTINGS) * len(C.ALL_MODELS) * len(C.SEEDS))
    out += [f"datasets present: {ds_seen}",
            f"seeds present:    {seeds_seen}",
            f"runs found:       {len(df)}",
            f"runs expected:    {exp}  (for these datasets/seeds)",
            f"full grid would be {full} runs"]
    have = set(zip(df.dataset, df.balance, df.features, df.model, df.seed))
    miss = [x for x in itertools.product(
        ds_seen, C.BALANCE_SETTINGS, C.FEATURE_SETTINGS,
        C.ALL_MODELS, seeds_seen) if x not in have]
    if miss:
        by_model = {}
        for m in miss:
            by_model[m[3]] = by_model.get(m[3], 0) + 1
        out += [f"INCOMPLETE: {len(miss)} of {exp} runs missing",
                f"   by model: {dict(sorted(by_model.items(), key=lambda z: -z[1]))}",
                "   first 10:"] + [f"      {m}" for m in miss[:10]]
        if len(set(m[3] for m in miss)) <= len(C.DL_MODELS) and \
           all(m[3] in C.DL_MODELS for m in miss):
            out += ["   NOTE: every missing run is a DL model. Either the job is",
                    "         still running, or DL is failing. Check the task log."]
    else:
        out += ["COMPLETE for the datasets and seeds present"]
    out.append("")

    # 1. F1 must be the harmonic mean of precision and recall
    p, r, f = df.precision, df.recall, df.f1
    expect = np.where((p + r) == 0, 0, 2 * p * r / (p + r))
    bad = df[np.abs(expect - f) > 1e-6]
    out += [f"[1] F1 == harmonic(P,R):        "
            f"{'PASS' if bad.empty else f'FAIL on {len(bad)} runs'}"]

    # 2. no two reported metrics may be identical across the board
    #     (this is the check that would have caught Tables 7 and 9)
    dup = []
    cols = ["accuracy", "precision", "recall", "f1", "f1_macro",
            "balanced_accuracy", "roc_auc", "pr_auc"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if np.allclose(df[a], df[b], atol=1e-9):
                dup.append(f"{a} == {b}")
    out += [f"[2] no duplicated metric columns: "
            f"{'PASS' if not dup else 'FAIL -> ' + ', '.join(dup)}"]

    # 3. ML and DL must use the same metric definition -> both families must
    #    satisfy every identity above, checked separately
    for fam in ["ML", "DL"]:
        d = df[df.family == fam]
        if d.empty:
            continue
        pp, rr, ff = d.precision, d.recall, d.f1
        ee = np.where((pp + rr) == 0, 0, 2 * pp * rr / (pp + rr))
        ok = np.abs(ee - ff).max() < 1e-6
        out += [f"[3] {fam} metric identity:         {'PASS' if ok else 'FAIL'}"]

    # 4. every configuration must have all seeds
    short = s[s[f"{METRICS[0]}_count"] != len(C.SEEDS)]
    out += [f"[4] all seeds present:            "
            f"{'PASS' if short.empty else f'FAIL on {len(short)} configs'}"]

    # 5. AUC ceiling warning - motivates reporting effect sizes, not just p
    for fam in ["ML", "DL"]:
        d = df[df.family == fam]
        if d.empty:
            continue
        rng = d.roc_auc.max() - d.roc_auc.min()
        flag = " <-- CEILING, report effect sizes not just p-values" if rng < 0.05 else ""
        out += [f"[5] {fam} ROC-AUC range:           {rng:.4f}{flag}"]

    # 6. identifier ablation delta, if present
    if "HIKARI-2021-noID" in set(df.dataset):
        a = df[df.dataset == "HIKARI-2021"].groupby("model").f1.mean()
        b = df[df.dataset == "HIKARI-2021-noID"].groupby("model").f1.mean()
        out += ["", "[6] HIKARI-2021 identifier-removal ablation (mean F1):"]
        for m in a.index:
            if m in b.index:
                out.append(f"       {m:<10} with={a[m]*100:6.2f}  "
                           f"without={b[m]*100:6.2f}  delta={(a[m]-b[m])*100:+6.2f} pp")
    out.append("")
    return "\n".join(out)


def main():
    df = load()
    df.to_csv("results_tidy.csv", index=False)
    s = summarise(df)
    s.to_csv("results_summary.csv", index=False)
    manuscript_tables(s, Path("tables"))
    rep = audit(df, s)
    Path("audit_report.txt").write_text(rep)
    print(rep)
    print("wrote results_tidy.csv, results_summary.csv, audit_report.txt")


if __name__ == "__main__":
    main()

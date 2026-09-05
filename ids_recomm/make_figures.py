#!/usr/bin/env python3
"""
Regenerate Figures 2-8 for the IDS-ReComm revision from results_tidy.csv.

    python make_figures.py --input results_tidy.csv --outdir figures

Produces, matching the manuscript's existing captions:

    Figure_02_accuracy.png   accuracy   (a) ML imbal (b) ML bal (c) DL imbal (d) DL bal
    Figure_03_recall.png     recall            "
    Figure_04_precision.png  precision         "
    Figure_05_f1.png         F1-score          "
    Figure_06_roc_auc.png    ROC-AUC           "
    Figure_07_time_ml.png    train vs test time, ML, all features, per dataset
    Figure_08_time_dl.png    train vs test time, DL, all features, per dataset

    Figure_06b_pr_auc.png    NEW - PR-AUC, same layout as Figure 6 (Reviewer C4)
    Figure_11_ablation.png   NEW - HIKARI-2021 identifier ablation (Reviewer C4)
    Figure_12_acc_vs_f1.png  NEW - accuracy/F1 divergence by dataset (Reviewer C5)

Bars show the mean over 5 seeds; error bars show one standard deviation, which
the submitted figures could not display because they came from single runs.

Figures 9 and 10 are schematic diagrams of IDS-ReComm, not plots, and are
unchanged by the re-run.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS2017", "HIKARI-2021"]
ML = ["DT", "RF", "GNB", "KNN", "SVM"]
DL = ["GRU", "LSTM", "MLP", "SimpleRNN", "CNN"]
FEATS = ["all", "top20", "top10"]
FEAT_LABEL = {"all": "All features", "top20": "Top-20", "top10": "Top-10"}
DS_COLOR = {"NSL-KDD": "#4E79A7", "UNSW-NB15": "#F28E2B",
            "CIC-IDS2017": "#59A14F", "HIKARI-2021": "#E15759"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.3,
    "grid.linestyle": "-", "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.bbox": "tight",
})


def panel(ax, df, models, metric, title, ylabel, ylim=None):
    """Grouped bars: x = model x feature setting, one bar per dataset."""
    xlabels, groups = [], []
    short = {"all": "All", "top20": "T20", "top10": "T10"}
    for mdl in models:
        for f in FEATS:
            xlabels.append(short[f])
            groups.append((mdl, f))

    n = len(DATASETS)
    width = 0.8 / n
    x = np.arange(len(groups))

    for i, ds in enumerate(DATASETS):
        means, sds = [], []
        for mdl, f in groups:
            r = df[(df.dataset == ds) & (df.model == mdl) & (df.features == f)]
            means.append(r[metric].mean() * 100 if len(r) else np.nan)
            sds.append(r[metric].std(ddof=1) * 100 if len(r) > 1 else 0.0)
        ax.bar(x + (i - (n - 1) / 2) * width, means, width,
               yerr=sds, capsize=1.5, label=ds, color=DS_COLOR[ds],
               edgecolor="white", linewidth=0.3,
               error_kw=dict(elinewidth=0.6, ecolor="#333333"))

    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=4)
    lo, hi = ylim if ylim else (0, 105)
    ax.set_ylim(lo, hi)
    # second-level x labels: model name centred under its three feature bars
    for gi, mdl in enumerate(models):
        ax.text(gi * 3 + 1, lo - (hi - lo) * 0.105, mdl, ha="center", va="top",
                fontsize=7.5, fontweight="bold", clip_on=False)
    for xv in range(3, len(groups), 3):
        ax.axvline(xv - 0.5, color="#cccccc", lw=0.5, zorder=0)


def four_panel(df, metric, label, fname, outdir, ylim=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    spec = [("(a) Machine learning, imbalanced", ML, "imbalanced"),
            ("(b) Machine learning, balanced", ML, "balanced"),
            ("(c) Deep learning, imbalanced", DL, "imbalanced"),
            ("(d) Deep learning, balanced", DL, "balanced")]
    for ax, (title, models, bal) in zip(axes.ravel(), spec):
        panel(ax, df[df.balance == bal], models, metric, title, label, ylim)
    axes[0, 0].legend(ncol=4, fontsize=8, frameon=False,
                      loc="upper center", bbox_to_anchor=(1.05, 1.20))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / fname)
    plt.close(fig)
    print(f"  wrote {fname}")


def timing(df, models, fname, outdir, title_family):
    """Train vs test time, all-features configuration, one panel per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    tags = "abcd"
    for ax, ds, tag in zip(axes.ravel(), DATASETS, tags):
        s = df[(df.dataset == ds) & (df.features == "all")]
        tr = [s[s.model == m].train_seconds.mean() for m in models]
        te = [s[s.model == m].predict_seconds.mean() for m in models]
        x = np.arange(len(models))
        ax.bar(x - 0.2, tr, 0.4, label="Training", color="#4E79A7")
        ax.bar(x + 0.2, te, 0.4, label="Inference", color="#F28E2B")
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylabel("Seconds (log scale)", fontsize=8)
        ax.set_title(f"({tag}) {ds}", fontsize=9, loc="left", pad=4)
        for xi, (a, b) in enumerate(zip(tr, te)):
            if a > 0: ax.text(xi - 0.2, a * 1.15, f"{a:.2f}", ha="center", fontsize=5)
            if b > 0: ax.text(xi + 0.2, b * 1.15, f"{b:.2f}", ha="center", fontsize=5)
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False,
                      loc="upper center", bbox_to_anchor=(1.05, 1.22))
    fig.suptitle(f"{title_family}: training and inference time, all-features "
                 f"configuration (CPU only)", fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / fname)
    plt.close(fig)
    print(f"  wrote {fname}")


def ablation(df, outdir):
    """HIKARI-2021 with vs without the four network identifier columns."""
    a = df[df.dataset == "HIKARI-2021"].groupby("model").f1.mean() * 100
    b = df[df.dataset == "HIKARI-2021-noID"].groupby("model").f1.mean() * 100
    models = [m for m in ML + DL if m in a.index and m in b.index]
    models.sort(key=lambda m: -(a[m] - b[m]))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(models))
    ax.bar(x - 0.2, [a[m] for m in models], 0.4,
           label="With identifiers", color="#4E79A7")
    ax.bar(x + 0.2, [b[m] for m in models], 0.4,
           label="Identifiers removed", color="#B07AA1")
    for i, m in enumerate(models):
        dlt = a[m] - b[m]
        ax.text(i, max(a[m], b[m]) + 2.5, f"{dlt:+.2f}", ha="center",
                fontsize=7, color="#C00000" if abs(dlt) > 1 else "#333333",
                fontweight="bold" if abs(dlt) > 1 else "normal")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Mean F1-score (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("HIKARI-2021 identifier-removal ablation (identical row set; "
                 "labels show change in percentage points)", fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(outdir / "Figure_11_ablation.png"); plt.close(fig)
    print("  wrote Figure_11_ablation.png")


def acc_vs_f1(df, outdir):
    """Accuracy-minus-F1 gap per dataset: the evaluation-distortion figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, fam, models in zip(axes, ["ML", "DL"], [ML, DL]):
        s = df[df.model.isin(models) & df.dataset.isin(DATASETS)]
        acc = s.groupby("dataset").accuracy.mean() * 100
        f1 = s.groupby("dataset").f1.mean() * 100
        x = np.arange(len(DATASETS))
        ax.bar(x - 0.2, [acc[d] for d in DATASETS], 0.4,
               label="Accuracy", color="#76B7B2")
        ax.bar(x + 0.2, [f1[d] for d in DATASETS], 0.4,
               label="F1-score", color="#E15759")
        for i, dset in enumerate(DATASETS):
            g = acc[dset] - f1[dset]
            ax.text(i, max(acc[dset], f1[dset]) + 2, f"{g:+.1f}", ha="center",
                    fontsize=8, fontweight="bold" if g > 10 else "normal",
                    color="#C00000" if g > 10 else "#333333")
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=7.5, rotation=12)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Mean score (%)", fontsize=8)
        ax.set_title(f"({'ab'[fam == 'DL']}) "
                     f"{'Machine' if fam == 'ML' else 'Deep'} learning",
                     fontsize=9, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Divergence between accuracy and F1-score by dataset "
                 "(labels show accuracy minus F1, percentage points)",
                 fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(outdir / "Figure_12_acc_vs_f1.png"); plt.close(fig)
    print("  wrote Figure_12_acc_vs_f1.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results_tidy.csv")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input)
    print(f"loaded {len(df)} runs from {a.input}")

    core = df[df.dataset.isin(DATASETS)]

    four_panel(core, "accuracy",  "Accuracy (%)",  "Figure_02_accuracy.png",  outdir)
    four_panel(core, "recall",    "Recall (%)",    "Figure_03_recall.png",    outdir)
    four_panel(core, "precision", "Precision (%)", "Figure_04_precision.png", outdir)
    four_panel(core, "f1",        "F1-score (%)",  "Figure_05_f1.png",        outdir)
    four_panel(core, "roc_auc",   "ROC-AUC (%)",   "Figure_06_roc_auc.png",   outdir,
               ylim=(50, 102))
    four_panel(core, "pr_auc",    "PR-AUC (%)",    "Figure_06b_pr_auc.png",   outdir,
               ylim=(0, 105))
    timing(core, ML, "Figure_07_time_ml.png", outdir, "Machine learning models")
    timing(core, DL, "Figure_08_time_dl.png", outdir, "Deep learning models")
    ablation(df, outdir)
    acc_vs_f1(core, outdir)

    print(f"\nall figures written to {outdir}/")
    print("Figures 9 and 10 (IDS-ReComm schematics) are unchanged.")


if __name__ == "__main__":
    main()

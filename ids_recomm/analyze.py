"""
Stage 5 - the statistical analysis Reviewer C asked for.

    python analyze.py --metric f1

What changed relative to the original analysis, and why.

The original ran one-way ANOVA treating every model-configuration result as an
independent observation. They are not independent: the same ten models are
re-measured on all four datasets, so observations are paired within model. That
inflates the F statistic, which is the most likely reason a design with 24
experiments produced p ~ 1.2e-18.

This script instead:
  * runs repeated-measures ANOVA with model as the subject and dataset as the
    within-subject factor;
  * reports partial eta-squared and omega-squared next to every F, so a
    significant result can be read against its actual magnitude;
  * gives bootstrap 95% CIs on each dataset's mean and on every pairwise
    difference;
  * applies Holm correction to the pairwise family, as before;
  * flags any comparison that is significant but whose CI lies inside a
    negligible window - the "statistically significant, operationally
    irrelevant" case section 3.5 of the paper already worries about.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config as C

NEGLIGIBLE_PP = 1.0   # differences under 1 percentage point flagged as trivial


def eta_omega(groups):
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_t = ss_b + ss_w
    df_b, df_w = k - 1, n - k
    ms_w = ss_w / df_w if df_w else np.nan
    eta2 = ss_b / ss_t if ss_t else np.nan
    omega2 = (ss_b - df_b * ms_w) / (ss_t + ms_w) if ss_t + ms_w else np.nan
    return float(eta2), float(max(omega2, 0.0))


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_ci(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = (rng.choice(a, (n, len(a)), replace=True).mean(1)
         - rng.choice(b, (n, len(b)), replace=True).mean(1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj


def analyse(df, metric, family, balance):
    d = df[(df.family == family) & (df.balance == balance)]
    if d.empty:
        return None
    datasets = sorted(set(d.dataset))
    groups = [d[d.dataset == ds][metric].values for ds in datasets]

    F, p = stats.f_oneway(*groups)
    eta2, omega2 = eta_omega(groups)

    rm = None
    try:
        from statsmodels.stats.anova import AnovaRM
        # subject = model x features x seed; within-subject factor = dataset
        d2 = d.copy()
        d2["subject"] = (d2.model + "|" + d2.features + "|" + d2.seed.astype(str))
        complete = (d2.groupby("subject").dataset.nunique() == len(datasets))
        d2 = d2[d2.subject.isin(complete[complete].index)]
        if not d2.empty:
            res = AnovaRM(d2, depvar=metric, subject="subject",
                          within=["dataset"], aggregate_func="mean").fit()
            t = res.anova_table
            rm = dict(F=float(t["F Value"].iloc[0]),
                      p=float(t["Pr > F"].iloc[0]),
                      df_num=float(t["Num DF"].iloc[0]),
                      df_den=float(t["Den DF"].iloc[0]),
                      n_subjects=int(d2.subject.nunique()))
    except Exception as e:  # statsmodels absent or design unbalanced
        rm = dict(error=str(e))

    means = {ds: dict(mean=float(g.mean()), sd=float(g.std(ddof=1)),
                      n=int(len(g)), ci95=boot_ci(g))
             for ds, g in zip(datasets, groups)}

    pairs, raw_p = [], []
    for a, b in itertools.combinations(datasets, 2):
        ga = d[d.dataset == a][metric].values
        gb = d[d.dataset == b][metric].values
        t, pp = stats.ttest_ind(ga, gb, equal_var=False)
        lo, hi = boot_diff_ci(ga, gb)
        pooled = np.sqrt((ga.var(ddof=1) + gb.var(ddof=1)) / 2)
        pairs.append(dict(a=a, b=b, t=float(t), p_raw=float(pp),
                          diff_pp=float((ga.mean() - gb.mean()) * 100),
                          diff_ci95_pp=[lo * 100, hi * 100],
                          cohens_d=float((ga.mean() - gb.mean()) / pooled) if pooled else None))
        raw_p.append(pp)
    for pr, adj in zip(pairs, holm(np.array(raw_p))):
        pr["p_holm"] = float(adj)
        pr["significant"] = bool(adj < 0.05)
        # negligibility is only meaningful for rate metrics
        pr["negligible"] = bool(max(abs(pr["diff_ci95_pp"][0]),
                                    abs(pr["diff_ci95_pp"][1])) < NEGLIGIBLE_PP) \
            if not metric.endswith("_seconds") else False
        pr["flag"] = ("significant but negligible"
                      if pr["significant"] and pr["negligible"] else "")

    return dict(metric=metric, family=family, balance=balance,
                oneway=dict(F=float(F), p=float(p), eta_squared=eta2,
                            omega_squared=omega2),
                repeated_measures=rm, per_dataset=means, pairwise=pairs)


def render(res) -> str:
    # Rate metrics are reported in percentage points; time metrics are seconds.
    # Scaling seconds by 100 (and labelling them "pp") was a display bug.
    is_time = res["metric"].endswith("_seconds")
    k = 1.0 if is_time else 100.0
    u = "s" if is_time else "pp"
    L = [f"### {res['family']} models, {res['balance']}, metric = {res['metric']}", ""]
    ow, rm = res["oneway"], res["repeated_measures"]
    L.append(f"one-way ANOVA (original method):  F = {ow['F']:.3f}, p = {ow['p']:.3e}, "
             f"eta^2 = {ow['eta_squared']:.3f}, omega^2 = {ow['omega_squared']:.3f}")
    if rm and "F" in rm:
        L.append(f"repeated-measures ANOVA:          F({rm['df_num']:.0f},{rm['df_den']:.0f}) "
                 f"= {rm['F']:.3f}, p = {rm['p']:.3e}, subjects = {rm['n_subjects']}")
    elif rm:
        L.append(f"repeated-measures ANOVA:          unavailable ({rm.get('error')})")
    L += ["", "per dataset (mean, bootstrap 95% CI):"]
    for ds, m in res["per_dataset"].items():
        L.append(f"   {ds:<20} {m['mean']*k:10.3f}{u}  "
                 f"[{m['ci95'][0]*k:9.3f}, {m['ci95'][1]*k:9.3f}]  n={m['n']}")
    L += ["", "pairwise (Holm-corrected):"]
    for pr in res["pairwise"]:
        star = "*" if pr["significant"] else " "
        s = k / 100.0   # diff_pp was stored already multiplied by 100
        L.append(f"  {star} {pr['a']:<18} vs {pr['b']:<18} "
                 f"diff = {pr['diff_pp']*s:+10.3f}{u} "
                 f"[{pr['diff_ci95_pp'][0]*s:+9.3f}, {pr['diff_ci95_pp'][1]*s:+9.3f}]  "
                 f"p_holm = {pr['p_holm']:.4g}  d = {pr['cohens_d']:+.2f}"
                 + (f"   <-- {pr['flag']}" if pr["flag"] else ""))
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="f1",
                    choices=["accuracy", "precision", "recall", "f1", "f1_macro",
                             "roc_auc", "pr_auc", "mcc",
                             "train_seconds", "predict_seconds"])
    ap.add_argument("--input", default="results_tidy.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    df = df[~df.dataset.str.endswith("-noID")]   # ablation analysed separately

    blocks, payload = [], []
    for fam in ["ML", "DL"]:
        for bal in C.BALANCE_SETTINGS:
            r = analyse(df, a.metric, fam, bal)
            if r:
                payload.append(r)
                blocks.append(render(r))

    txt = "\n".join(blocks)
    Path(f"stats_{a.metric}.txt").write_text(txt)
    Path(f"stats_{a.metric}.json").write_text(json.dumps(payload, indent=2))
    print(txt)


if __name__ == "__main__":
    main()

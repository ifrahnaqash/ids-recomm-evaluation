"""
Loading, cleaning, and the CORRECTED preprocessing pipeline.

The whole point of this file is the ordering. In the original notebooks the
order was:

    clean -> encode -> correlation filter -> RFE(full data + full labels)
          -> train_test_split -> scaler.fit_transform(train)
          -> scaler.fit_transform(test)   <-- refit on test
          -> SMOTE(train)                 <-- this part was correct

Here it is:

    clean -> train/test split -> train/val split
          -> OneHotEncoder.fit(train)     -> transform(val, test)
          -> correlation filter on TRAIN features only, label excluded
          -> RFE.fit(TRAIN only)          -> apply to val, test
          -> StandardScaler.fit(train)    -> transform(val, test)
          -> SMOTE(train only)            -> val and test remain untouched

Nothing downstream of the split ever sees a test row.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================================
# Stage 1 - raw -> cleaned parquet (run once per dataset)
# ==========================================================================
def build_clean_frame(dataset: str) -> pd.DataFrame:
    spec = C.DATASETS[dataset]

    # Ablation variants inherit the parent's already-cleaned rows and simply
    # drop columns. No second deduplication: the comparison must isolate the
    # effect of the columns, not of a different row set.
    parent = spec.get("derive_from")
    if parent:
        src = cache_path(parent)
        if not src.exists():
            raise FileNotFoundError(
                f"{dataset} derives from {parent}, but {src} does not exist. "
                f"Run: python prepare.py --dataset '{parent}' --stage clean")
        df = pd.read_parquet(src)
        extra = [c for c in spec["drop_cols"] if c in df.columns]
        out = df.drop(columns=extra)
        print(f"  derived from {parent}: {len(out):,} rows (unchanged), "
              f"dropped {len(extra)} column(s): {extra}")
        return out

    frames = []
    for f in spec["files"]:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"{dataset}: missing raw file {f}")
        kw = dict(spec["read_kwargs"])
        if spec["column_names"]:
            kw["names"] = spec["column_names"]
        frames.append(pd.read_csv(f, **kw))
    df = pd.concat(frames, ignore_index=True)

    # normalise column names (CIC-IDS2017 CSVs carry leading spaces)
    df.columns = [str(c).strip() for c in df.columns]

    label_col = spec["label_col"]
    if label_col not in df.columns:
        raise KeyError(
            f"{dataset}: label column {label_col!r} not found. "
            f"Columns are: {list(df.columns)[:40]}"
        )

    # binary target: 0 = benign, 1 = attack
    lab = df[label_col].astype(str).str.strip().str.lower()
    benign = {b.lower() for b in spec["benign_values"]}
    y = (~lab.isin(benign)).astype(np.int8)

    drop = [c for c in spec["drop_cols"] if c in df.columns] + [label_col]
    X = df.drop(columns=drop)

    # cleaning, exactly as in the original notebooks
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    X, y = X[keep], y[keep]

    # NOTE: original used drop_duplicates(keep=False), which removes *every*
    # copy including the first. keep="first" is the standard reading of
    # "duplicates removed" and is what the manuscript text describes.
    #
    # Deduplication is on FEATURES ONLY. That is deliberate and it matters:
    # the original notebooks deduplicated while `id` (UNSW-NB15) and `uid`
    # (HIKARI-2021) were still present. Those are unique per row, so the check
    # could never fire - HIKARI's notebook duly printed "Number of duplicate
    # rows: 0". With the identifiers gone the duplicates become visible.
    feat_dup = int(X.duplicated().sum())
    full_dup = int(pd.concat([X, y.rename("__y__")], axis=1).duplicated().sum())
    contradictory = feat_dup - full_dup   # same features, disagreeing labels

    n_before = len(X)
    dropped_contradictory = 0
    if C.CONTRADICTION_POLICY == "drop_all" and contradictory > 0:
        # Identify every feature-group that carries more than one label and
        # remove ALL of its members. Done on the duplicated subset only, so the
        # groupby never touches the (much larger) set of unique rows.
        dup_mask = X.duplicated(keep=False)
        if dup_mask.any():
            sub = X[dup_mask].copy()
            sub["__y__"] = y[dup_mask].values
            key = pd.util.hash_pandas_object(X[dup_mask], index=False)
            sub["__k__"] = key.values
            nlab = sub.groupby("__k__")["__y__"].transform("nunique")
            bad_keys = set(sub.loc[nlab > 1, "__k__"].unique())
            if bad_keys:
                full_key = pd.util.hash_pandas_object(X, index=False)
                drop_rows = full_key.isin(bad_keys).values
                dropped_contradictory = int(drop_rows.sum())
                X, y = X[~drop_rows], y[~drop_rows]

    dup = X.duplicated(keep="first")
    X, y = X[~dup], y[~dup]

    print(f"  duplicates: {feat_dup:,} feature-identical "
          f"({full_dup:,} with matching labels, "
          f"{contradictory:,} with contradictory labels)")
    print(f"  contradiction policy '{C.CONTRADICTION_POLICY}': "
          f"removed {dropped_contradictory:,} rows in ambiguous feature-groups; "
          f"{n_before:,} -> {len(X):,} rows")

    out = X.reset_index(drop=True)
    out["__label__"] = y.reset_index(drop=True).values
    out.attrs["dup_stats"] = dict(feature_duplicates=feat_dup,
                                  full_duplicates=full_dup,
                                  contradictory=contradictory,
                                  contradiction_policy=C.CONTRADICTION_POLICY,
                                  rows_dropped_contradictory=dropped_contradictory)
    return out


def cache_path(dataset: str) -> Path:
    return C.CACHE_DIR / f"clean_{dataset}.parquet"


def prep_path(dataset: str, seed: int) -> Path:
    return C.CACHE_DIR / f"prep_{dataset}_seed{seed}.joblib"


# ==========================================================================
# Stage 2 - per (dataset, seed): split, fit encoder / corr filter / RFE / scalers
# ==========================================================================
def _split(df: pd.DataFrame, seed: int):
    y = df["__label__"].values
    X = df.drop(columns="__label__")
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full,
        test_size=C.VAL_SIZE_OF_TRAIN, random_state=seed, stratify=y_tr_full
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _encode_fit(X_tr: pd.DataFrame):
    """Fit encoders on TRAINING rows only.

    Cardinality matters here. HIKARI-2021's `originh` has 2,899 distinct values
    and `responh` has 7,991; one-hot encoding them would produce ~10,900 dense
    columns over 555k rows. Columns above CARDINALITY_MAX therefore get ordinal
    encoding (which is what the original HIKARI notebook did via LabelEncoder),
    and everything below it gets one-hot (which is what the original NSL-KDD
    notebook did via get_dummies). The threshold makes that split explicit and
    uniform instead of differing silently per dataset.

    Note: test `is_numeric_dtype`, not `dtype == object`. After a parquet
    round-trip, string columns come back as arrow-backed dtypes, not object.
    """
    num = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    cat_all = [c for c in X_tr.columns if c not in num]
    hi = [c for c in cat_all if X_tr[c].nunique() > C.CARDINALITY_MAX]
    cat = [c for c in cat_all if c not in hi]

    enc = None
    if cat:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            drop="first", dtype=np.float32)
        enc.fit(X_tr[cat])

    ordmap = {}
    for c in hi:
        # categories sorted, fitted on train; unseen test values -> -1
        cats = pd.Index(sorted(X_tr[c].astype(str).unique()))
        ordmap[c] = {v: i for i, v in enumerate(cats)}
    return (enc, ordmap), cat, num


def _encode_apply(X, encoder, cat, num) -> tuple[np.ndarray, list[str]]:
    enc, ordmap = encoder
    blocks, names = [], []
    if num:
        blocks.append(X[num].to_numpy(dtype=np.float32))
        names += list(num)
    if enc is not None and cat:
        blocks.append(enc.transform(X[cat]))
        names += list(enc.get_feature_names_out(cat))
    for c, m in ordmap.items():
        col = X[c].astype(str).map(m).fillna(-1).to_numpy(dtype=np.float32)
        blocks.append(col.reshape(-1, 1))
        names.append(f"{c}__ordinal")
    return np.hstack(blocks), names


def _corr_drop(M: np.ndarray, names: list[str], thresh: float) -> list[int]:
    """Correlation filter on TRAIN features only. Label is never included."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(M, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    drop, kept = [], []
    for j in range(corr.shape[0]):
        if any(corr[j, k] > thresh for k in kept):
            drop.append(j)
        else:
            kept.append(j)
    return drop


def build_prep(dataset: str, seed: int, n_jobs: int = -1) -> dict:
    """Fit every transform on training data only; persist for the run stage."""
    df = pd.read_parquet(cache_path(dataset))
    X_tr, X_va, X_te, y_tr, y_va, y_te = _split(df, seed)

    enc, cat, num = _encode_fit(X_tr)
    M_tr, names = _encode_apply(X_tr, enc, cat, num)

    # constant columns carry no information and destabilise the corr matrix
    const = [j for j in range(M_tr.shape[1]) if np.ptp(M_tr[:, j]) == 0]
    dropped = _corr_drop(M_tr, names, C.CORR_THRESHOLD)
    drop_idx = sorted(set(dropped) | set(const))
    keep_idx = [j for j in range(M_tr.shape[1]) if j not in set(drop_idx)]
    M_tr = M_tr[:, keep_idx]
    kept_names = [names[j] for j in keep_idx]

    # RFE fitted on TRAIN ONLY. "all" means all surviving features - no RFE.
    #
    # The two selections are nested: top20 is chosen from all features, then
    # top10 is chosen from those 20. Two benefits over fitting each from
    # scratch. It is much cheaper - the second elimination runs on 20 columns
    # instead of ~80, which matters when the training split is 1.98M rows. And
    # it guarantees top10 is a subset of top20, so the three feature settings
    # form a proper nested sequence rather than three unrelated subsets, which
    # is what the manuscript's "top-20 / top-10" framing implies.
    selections = {"all": list(range(len(kept_names)))}

    k20 = min(20, M_tr.shape[1])
    rfe20 = RFE(RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
                n_features_to_select=k20, step=0.5)
    rfe20.fit(M_tr, y_tr)
    idx20 = list(np.where(rfe20.support_)[0].astype(int))
    selections["top20"] = idx20

    k10 = min(10, len(idx20))
    rfe10 = RFE(RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
                n_features_to_select=k10, step=1)
    rfe10.fit(M_tr[:, idx20], y_tr)
    selections["top10"] = [idx20[j] for j in np.where(rfe10.support_)[0]]

    # one scaler per feature setting, fitted on train rows of that subset
    scalers = {}
    for tag, idx in selections.items():
        sc = StandardScaler().fit(M_tr[:, idx])
        scalers[tag] = sc

    prep = dict(
        dataset=dataset, seed=seed, encoder=enc, cat_cols=cat, num_cols=num,
        encoded_names=names, keep_idx=keep_idx, kept_names=kept_names,
        corr_dropped=[names[j] for j in dropped],
        constant_dropped=[names[j] for j in const],
        selections=selections, scalers=scalers,
        selected_names={t: [kept_names[j] for j in idx]
                        for t, idx in selections.items()},
        n_train=len(y_tr), n_val=len(y_va), n_test=len(y_te),
        pos_rate_train=float(y_tr.mean()), pos_rate_test=float(y_te.mean()),
    )
    joblib.dump(prep, prep_path(dataset, seed), compress=3)

    (C.CACHE_DIR / f"features_{dataset}_seed{seed}.json").write_text(json.dumps(
        {k: prep["selected_names"][k] for k in prep["selections"]}, indent=2))
    return prep


# ==========================================================================
# Stage 3 - materialise matrices for one run
# ==========================================================================
def materialise(dataset: str, seed: int, features: str, balance: str):
    """Return scaled (X_tr, y_tr, X_va, y_va, X_te, y_te) for one configuration."""
    from imblearn.over_sampling import SMOTE

    df = pd.read_parquet(cache_path(dataset))
    prep = joblib.load(prep_path(dataset, seed))
    X_tr, X_va, X_te, y_tr, y_va, y_te = _split(df, seed)

    def to_matrix(X):
        M, _ = _encode_apply(X, prep["encoder"], prep["cat_cols"], prep["num_cols"])
        M = M[:, prep["keep_idx"]]
        return M[:, prep["selections"][features]]

    A_tr, A_va, A_te = to_matrix(X_tr), to_matrix(X_va), to_matrix(X_te)

    sc = prep["scalers"][features]
    A_tr = sc.transform(A_tr)
    A_va = sc.transform(A_va)   # transform, NOT fit_transform  <-- bug fix
    A_te = sc.transform(A_te)   # transform, NOT fit_transform  <-- bug fix

    if balance == "balanced":
        sm = SMOTE(sampling_strategy="minority", random_state=seed)
        A_tr, y_tr = sm.fit_resample(A_tr, y_tr)   # train only; val/test untouched

    return (A_tr.astype(np.float32), y_tr,
            A_va.astype(np.float32), y_va,
            A_te.astype(np.float32), y_te,
            prep["selected_names"][features])

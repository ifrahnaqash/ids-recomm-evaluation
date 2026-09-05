"""
Batched runner for clusters that allow only one concurrent job per user.

    python run_group.py --dataset CIC-IDS2017 --seed 42 --balance balanced

The original design put one (dataset, seed, features, balance, model) per SLURM
array task: 1500 tasks. That is correct when tasks run in parallel. Under
MaxJobsPU=1 they run serially, and each task pays the full cost of re-reading
the parquet, re-encoding, re-splitting, re-scaling and re-SMOTEing before it
trains anything. On CIC-IDS2017 that setup cost is minutes, so 1500 tasks would
burn more wall-clock on repeated preprocessing than on the models themselves.

This runner takes one (dataset, seed, balance) group and loops over the three
feature settings and ten models inside a single process. Data is materialised
three times per task instead of thirty, and the interpreter, imports and
TensorFlow initialisation are paid once.

Grid: 5 datasets x 5 seeds x 2 balance settings = 50 tasks.

Output format is byte-identical to run_experiment.py, so aggregate.py and
analyze.py are unchanged. Individual runs are still skipped if their result
JSON exists, so an interrupted task resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from datetime import datetime, timezone

import numpy as np

import config as C
import data as D
import models as M
from run_experiment import tag_for


def one_run(dataset, seed, features, balance, model, mats, force=False):
    """Train and evaluate a single model against already-materialised matrices."""
    tag = tag_for(dataset, seed, features, balance, model)
    out = C.RESULT_DIR / f"{tag}.json"
    if out.exists() and not force:
        print(f"  [skip] {model}", flush=True)
        return "skipped"

    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names, load_s = mats
    n_train_used = len(y_tr)
    epochs_run = None

    if model in C.ML_MODELS:
        Xf, yf = X_tr, y_tr
        if model == "SVM":
            Xf, yf, n_train_used = M.subsample_for_svm(X_tr, y_tr, seed)
        clf = M.build_ml(model, seed)
        t0 = time.time(); clf.fit(Xf, yf); train_s = time.time() - t0
        t0 = time.time(); y_pred = clf.predict(X_te); predict_s = time.time() - t0
        y_prob = clf.predict_proba(X_te)[:, 1]
    else:
        import tensorflow as tf
        from tensorflow.keras import backend as K
        from tensorflow.keras.callbacks import EarlyStopping

        K.clear_session()   # otherwise ten models accumulate in one process
        net = M.build_dl(model, X_tr.shape[1], seed)
        R_tr, R_va, R_te = (M.reshape_for(model, X_tr),
                            M.reshape_for(model, X_va),
                            M.reshape_for(model, X_te))
        es = EarlyStopping(monitor="val_loss", patience=C.DL_PATIENCE,
                           min_delta=C.DL_MIN_DELTA,
                           restore_best_weights=True, verbose=0)
        t0 = time.time()
        hist = net.fit(R_tr, y_tr, validation_data=(R_va, y_va),
                       epochs=C.DL_MAX_EPOCHS, batch_size=C.DL_BATCH_SIZE,
                       callbacks=[es], verbose=0)
        train_s = time.time() - t0
        epochs_run = len(hist.history["loss"])
        t0 = time.time()
        y_prob = net.predict(R_te, batch_size=C.DL_BATCH_SIZE, verbose=0).ravel()
        predict_s = time.time() - t0
        y_pred = (y_prob >= 0.5).astype(np.int8)
        del net
        K.clear_session()

    metrics = M.compute_metrics(y_te, y_pred, y_prob)
    provenance = dict(
        tag=tag, dataset=dataset, seed=seed, features=features,
        balance=balance, model=model,
        family="ML" if model in C.ML_MODELS else "DL",
        n_features=int(X_tr.shape[1]), feature_names=feat_names,
        n_train=int(len(y_tr)), n_train_used=int(n_train_used),
        n_val=int(len(y_va)),
        svm_cap=C.SVM_MAX_TRAIN if model == "SVM" else None,
        epochs_run=epochs_run, max_epochs=C.DL_MAX_EPOCHS,
        train_seconds=train_s, predict_seconds=predict_s,
        load_seconds=load_s, total_seconds=train_s + predict_s,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
    )
    out.write_text(json.dumps({**provenance, **metrics}, indent=2))
    np.savez_compressed(C.PRED_DIR / f"{tag}.npz",
                        y_true=np.asarray(y_te, dtype=np.int8),
                        y_pred=np.asarray(y_pred, dtype=np.int8),
                        y_prob=np.asarray(y_prob, dtype=np.float32))
    print(f"  [done] {model:<10} acc={metrics['accuracy']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"F1={metrics['f1']:.4f} PR={metrics['pr_auc']:.4f} "
          f"train={train_s:.1f}s predict={predict_s:.1f}s"
          + (f" epochs={epochs_run}" if epochs_run else ""), flush=True)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASET_NAMES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--balance", required=True, choices=C.BALANCE_SETTINGS)
    ap.add_argument("--features", nargs="*", default=C.FEATURE_SETTINGS)
    ap.add_argument("--models", nargs="*", default=C.ALL_MODELS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    t_group = time.time()
    print(f"=== {a.dataset} seed={a.seed} {a.balance} on {platform.node()} ===",
          flush=True)

    failures = []
    for features in a.features:
        # skip materialisation entirely if every model here is already done
        pending = [m for m in a.models
                   if a.force or not (C.RESULT_DIR /
                       f"{tag_for(a.dataset, a.seed, features, a.balance, m)}.json").exists()]
        if not pending:
            print(f"-- {features}: all {len(a.models)} models already done", flush=True)
            continue

        print(f"-- {features}: materialising ({len(pending)} pending)", flush=True)
        t0 = time.time()
        try:
            X_tr, y_tr, X_va, y_va, X_te, y_te, names = D.materialise(
                a.dataset, a.seed, features, a.balance)
        except Exception:
            print(f"   MATERIALISE FAILED for {features}:\n{traceback.format_exc()}",
                  flush=True)
            failures.append((features, "materialise"))
            continue
        load_s = time.time() - t0
        print(f"   {X_tr.shape[0]:,} train x {X_tr.shape[1]} features, "
              f"{X_te.shape[0]:,} test, pos_rate_train={y_tr.mean():.4f} "
              f"({load_s:.1f}s)", flush=True)
        mats = (X_tr, y_tr, X_va, y_va, X_te, y_te, names, load_s)

        for model in a.models:
            try:
                one_run(a.dataset, a.seed, features, a.balance, model, mats, a.force)
            except Exception:
                # one model failing must not lose the other nine
                print(f"  FAILED {model}:\n{traceback.format_exc()}", flush=True)
                failures.append((features, model))

        del X_tr, y_tr, X_va, y_va, X_te, y_te, mats

    mins = (time.time() - t_group) / 60
    print(f"=== group finished in {mins:.1f} min, {len(failures)} failure(s) ===",
          flush=True)
    if failures:
        for f in failures:
            print(f"    FAILED: {f}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

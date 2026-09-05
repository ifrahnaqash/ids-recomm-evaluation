"""
Stage 3 - one (dataset, seed, features, balance, model) configuration.

    python run_experiment.py --dataset NSL-KDD --seed 42 \
        --features all --balance balanced --model CNN

Writes:
    results/<tag>.json      metrics + timings + provenance
    predictions/<tag>.npz   y_true, y_pred, y_prob  (so statistics can be
                            recomputed later without retraining anything)

Idempotent: an existing result JSON is skipped unless --force is passed.
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np

import config as C
import data as D
import models as M


def tag_for(dataset, seed, features, balance, model):
    return f"{dataset}__{balance}__{features}__{model}__seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASET_NAMES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", required=True, choices=C.FEATURE_SETTINGS)
    ap.add_argument("--balance", required=True, choices=C.BALANCE_SETTINGS)
    ap.add_argument("--model", required=True, choices=C.ALL_MODELS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tag = tag_for(a.dataset, a.seed, a.features, a.balance, a.model)
    out = C.RESULT_DIR / f"{tag}.json"
    if out.exists() and not a.force:
        print(f"[skip] {tag}")
        return

    t_load = time.time()
    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names = D.materialise(
        a.dataset, a.seed, a.features, a.balance)
    load_s = time.time() - t_load

    n_train_used = len(y_tr)
    epochs_run = None

    if a.model in C.ML_MODELS:
        Xf, yf = X_tr, y_tr
        if a.model == "SVM":
            Xf, yf, n_train_used = M.subsample_for_svm(X_tr, y_tr, a.seed)
        clf = M.build_ml(a.model, a.seed)

        t0 = time.time(); clf.fit(Xf, yf); train_s = time.time() - t0
        t0 = time.time(); y_pred = clf.predict(X_te); predict_s = time.time() - t0
        y_prob = clf.predict_proba(X_te)[:, 1]

    else:
        from tensorflow.keras.callbacks import EarlyStopping

        net = M.build_dl(a.model, X_tr.shape[1], a.seed)
        R_tr = M.reshape_for(a.model, X_tr)
        R_va = M.reshape_for(a.model, X_va)
        R_te = M.reshape_for(a.model, X_te)

        # Uniform stopping rule across all datasets, monitored on the real
        # (never resampled) validation split. Replaces the original's
        # per-dataset epoch counts of 400 vs 8-10.
        es = EarlyStopping(monitor="val_loss", patience=C.DL_PATIENCE,
                           min_delta=C.DL_MIN_DELTA,
                           restore_best_weights=True, verbose=0)
        t0 = time.time()
        hist = net.fit(R_tr, y_tr, validation_data=(R_va, y_va),
                       epochs=C.DL_MAX_EPOCHS, batch_size=C.DL_BATCH_SIZE,
                       callbacks=[es], verbose=2)
        train_s = time.time() - t0
        epochs_run = len(hist.history["loss"])

        t0 = time.time()
        y_prob = net.predict(R_te, batch_size=C.DL_BATCH_SIZE, verbose=0).ravel()
        predict_s = time.time() - t0
        y_pred = (y_prob >= 0.5).astype(np.int8)

    metrics = M.compute_metrics(y_te, y_pred, y_prob)

    provenance = dict(
        tag=tag, dataset=a.dataset, seed=a.seed, features=a.features,
        balance=a.balance, model=a.model,
        family="ML" if a.model in C.ML_MODELS else "DL",
        n_features=int(X_tr.shape[1]), feature_names=feat_names,
        n_train=int(len(y_tr)), n_train_used=int(n_train_used),
        n_val=int(len(y_va)),
        svm_cap=C.SVM_MAX_TRAIN if a.model == "SVM" else None,
        epochs_run=epochs_run, max_epochs=C.DL_MAX_EPOCHS,
        train_seconds=train_s, predict_seconds=predict_s,
        load_seconds=load_s, total_seconds=train_s + predict_s,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
    )
    record = {**provenance, **metrics}   # metrics keys (n_test, pos_rate) win
    out.write_text(json.dumps(record, indent=2))
    np.savez_compressed(C.PRED_DIR / f"{tag}.npz",
                        y_true=np.asarray(y_te, dtype=np.int8),
                        y_pred=np.asarray(y_pred, dtype=np.int8),
                        y_prob=np.asarray(y_prob, dtype=np.float32))

    print(f"[done] {tag}  acc={metrics['accuracy']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"F1={metrics['f1']:.4f} ROC={metrics['roc_auc']:.4f} "
          f"PR={metrics['pr_auc']:.4f}  train={train_s:.1f}s")


if __name__ == "__main__":
    main()

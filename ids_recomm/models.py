"""
Model construction and the UNIFIED metric computation.

Two things this file fixes.

1. Metric definition. The original computed ML metrics with
   sklearn `average="weighted"` and DL metrics with custom Keras `f1_m`,
   `precision_m`, `recall_m` read out of `model.evaluate()`. Those are not the
   same quantity:
     - weighted-average recall over all classes is mathematically identical to
       accuracy, which is why Tables 7 and 9 of the results document came out
       byte-identical;
     - Keras custom metrics are averaged per batch, which is a biased estimator
       of the epoch-level score.
   Here every model - ML and DL alike - produces y_pred and y_prob, and every
   metric is computed by sklearn from those arrays. ML and DL numbers become
   directly comparable, which the paper's central claim requires.

2. Metric ordering. The original wrote ML results as
   [accuracy, recall, precision, f1] and DL results as
   [accuracy, f1, precision, recall] into columns
   ['Accuracy','Recall','Precision','F1-Score'], transposing recall and F1 for
   every DL row. Here metrics are returned in a dict, so ordering cannot slip.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as C


# ==========================================================================
# Metrics
# ==========================================================================
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Every metric for every model comes from here. No exceptions."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        # primary reporting metrics: positive (attack) class
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        # macro variants, reported alongside so both conventions are available
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),   # Reviewer C4
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n_test=int(len(y_true)), pos_rate=float(y_true.mean()),
    )
    # internal consistency guard: F1 must equal the harmonic mean of P and R.
    # This is exactly the check that exposed the original transposition.
    p, r, f = m["precision"], m["recall"], m["f1"]
    expect = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    m["f1_consistency_error"] = abs(expect - f)
    assert m["f1_consistency_error"] < 1e-6, "metric ordering corrupted"
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in m.items()}


# ==========================================================================
# Classical ML
# ==========================================================================
def build_ml(name: str, seed: int, n_jobs: int = -1):
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    if name == "GNB":
        return GaussianNB()
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=3, n_jobs=n_jobs)
    if name == "SVM":
        n_est = 10
        return BaggingClassifier(
            SVC(kernel="rbf", probability=True, class_weight="balanced",
                random_state=seed),
            max_samples=1.0 / n_est, n_estimators=n_est,
            n_jobs=n_jobs, random_state=seed)
    raise ValueError(name)


def subsample_for_svm(X, y, seed: int):
    """Uniform stratified cap on SVM training size. See config.SVM_MAX_TRAIN."""
    cap = C.SVM_MAX_TRAIN
    if cap is None or len(y) <= cap:
        return X, y, len(y)
    from sklearn.model_selection import train_test_split
    Xs, _, ys, _ = train_test_split(X, y, train_size=cap,
                                    random_state=seed, stratify=y)
    return Xs, ys, len(ys)


# ==========================================================================
# Deep learning
# ==========================================================================
def build_dl(name: str, n_features: int, seed: int):
    import tensorflow as tf
    from tensorflow.keras.layers import (GRU, LSTM, Conv1D, Dense, Dropout,
                                         Flatten, MaxPooling1D, SimpleRNN)
    from tensorflow.keras.models import Sequential

    tf.keras.utils.set_random_seed(seed)
    shape = (1, n_features)
    m = Sequential()

    if name == "GRU":
        m.add(GRU(20, return_sequences=True, input_shape=shape))
        m.add(Dropout(0.2)); m.add(GRU(20, return_sequences=False))
    elif name == "LSTM":
        m.add(LSTM(20, return_sequences=True, input_shape=shape))
        m.add(Dropout(0.2)); m.add(LSTM(20, return_sequences=False))
    elif name == "SimpleRNN":
        m.add(SimpleRNN(50, input_shape=shape)); m.add(Dense(50, activation="relu"))
    elif name == "MLP":
        m.add(Dense(20, activation="relu", input_shape=(n_features,)))
        m.add(Dense(20, activation="relu"))
    elif name == "CNN":
        m.add(Conv1D(32, 3, activation="relu", padding="same",
                     input_shape=(n_features, 1)))
        m.add(MaxPooling1D(2, padding="same"))
        m.add(Conv1D(64, 3, activation="relu", padding="same"))
        m.add(MaxPooling1D(2, padding="same"))
        m.add(Flatten()); m.add(Dense(128, activation="relu")); m.add(Dropout(0.5))
    else:
        raise ValueError(name)

    m.add(Dense(1, activation="sigmoid"))
    # NOTE: metrics=[] on purpose. Nothing is read out of model.evaluate().
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=[])
    return m


def reshape_for(name: str, X):
    if name == "MLP":
        return X
    if name == "CNN":
        return X.reshape(X.shape[0], X.shape[1], 1)
    return X.reshape(X.shape[0], 1, X.shape[1])   # GRU / LSTM / SimpleRNN

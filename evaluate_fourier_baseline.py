

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import im, tfreq, load_subject_data
from src.features_psd import channels, extract_psd_features, select_channels

log = logging.getLogger("baseline")


def build_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_validate(
        build_model(), X, y, groups=groups,
        cv=cv, scoring=["accuracy", "roc_auc"], return_train_score=True, n_jobs=-1,
)

    rows = []
    for i, (_, test_idx) in enumerate(cv.split(X, y, groups)):
        rows.append({
            "fold": i + 1,
            "n_test_subjects": len(np.unique(groups[test_idx])),
            "n_test_epochs": len(test_idx),
            "train_acc": scores["train_accuracy"][i],
            "test_acc": scores["test_accuracy"][i],
            "test_auc": scores["test_roc_auc"][i],
        })
    return pd.DataFrame(rows)


def report(name: str, df: pd.DataFrame, n_features: int) -> None:
    print(f"\n{name}  ({n_features} features)")
    print("-" * 68)
    print(df.to_string(
        index=False,
        formatters={"train_acc": "{:.4f}".format,
                    "test_acc": "{:.4f}".format,
                    "test_auc": "{:.4f}".format},
    ))
    print("-" * 68)
    print(f"mean test accuracy : {df.test_acc.mean():.4f} +/- {df.test_acc.std(ddof=1):.4f}")
    print(f"mean test roc auc  : {df.test_auc.mean():.4f} +/- {df.test_auc.std(ddof=1):.4f}")
    print(f"mean train accuracy: {df.train_acc.mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=20, help="number of subjects to load")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--runs", type=int, nargs="+", default=list(im))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    subject_ids = range(1, args.subjects + 1)
    X_epochs, y, groups, ch_names = load_subject_data(subject_ids, runs=args.runs)

    print(f"\nloaded {X_epochs.shape[0]} epochs from {len(np.unique(groups))} subjects "
          f"| {X_epochs.shape[1]} channels x {X_epochs.shape[2]} samples")
    print(f"class balance: left={int((y == 0).sum())} right={int((y == 1).sum())}")
    print(f"majority-class baseline: {max(np.bincount(y)) / len(y):.4f}")

    X_smc = extract_psd_features(
        select_channels(X_epochs, ch_names, channels), sfreq=tfreq
    )
    X_all = extract_psd_features(X_epochs, sfreq=tfreq)

    report(f"sensorimotor only {channels}",
           evaluate(X_smc, y, groups, args.n_splits), X_smc.shape[1])
    report("all channels",
           evaluate(X_all, y, groups, args.n_splits), X_all.shape[1])


if __name__ == "__main__":
    main()

"""stage 2 - fbcsp vs psd"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import all_subjects, im, load_subject_data, tfreq
from src.features_psd import channels, extract_psd_features, select_channels
from src.fbcsp import FilterBankCSP, apply_filter_bank, bank_4_32, bank_4_40, bank_mu_beta, build_fbcsp
from src.results import append_metrics

log = logging.getLogger("stage2")
RESULTS = Path(__file__).resolve().parents[1] / "results"


def psd_pipeline():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def score_split(model, X, y, tr, te):
    model.fit(X[tr], y[tr])
    p_tr, p_te = model.predict(X[tr]), model.predict(X[te])
    d_tr, d_te = model.decision_function(X[tr]), model.decision_function(X[te])
    return {
        "train_acc": accuracy_score(y[tr], p_tr),
        "test_acc": accuracy_score(y[te], p_te),
        "train_auc": roc_auc_score(y[tr], d_tr),
        "test_auc": roc_auc_score(y[te], d_te),
    }


def run_cv(name, factory, X, y, groups, splits, n_features):
    """Score one method across pre-computed splits. factory() returns a fresh pipeline."""
    rows = []
    t0 = time.perf_counter()
    for fold, (tr, te) in enumerate(splits, start=1):
        m = score_split(factory(), X, y, tr, te)
        rows.append({"method": name, "fold": fold, "n_features": n_features,
                     "n_test_subjects": len(np.unique(groups[te])),
                     "n_test_epochs": len(te), **m})
    log.info("%-28s done in %5.1fs  test_acc=%.4f", name,
             time.perf_counter() - t0, np.mean([r["test_acc"] for r in rows]))
    return rows


def summarise(df):
    g = df.groupby("method", sort=False)
    out = pd.DataFrame({
        "n_feat": g.n_features.first(),
        "train_acc": g.train_acc.mean(),
        "test_acc": g.test_acc.mean(),
        "test_sd": g.test_acc.std(ddof=1),
        "test_auc": g.test_auc.mean(),
    })
    out["gap"] = out.train_acc - out.test_acc
    return out.reset_index()


def plot_patterns(X_bank, y, ch_names, bank, band_idx, path):
    """CSP patterns for one sub-band, fit on all epochs -- inspection only, never scored."""
    info = mne.create_info(list(ch_names), tfreq, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    csp = FilterBankCSP(n_bands=len(bank)).fit(X_bank, y)
    lo, hi = bank[band_idx]
    fig = csp.csps_[band_idx].plot_patterns(info, ch_type="eeg", show=False,
                                            components=range(csp.n_components),
                                            units="a.u.", size=1.5)
    fig.suptitle(f"CSP patterns, {lo:.0f}-{hi:.0f} Hz (all subjects)", y=1.05)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--runs", type=int, nargs="+", default=list(im))
    ap.add_argument("--k", type=int, default=12, help="mutual-information features kept")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mne.set_log_level("ERROR")
    RESULTS.mkdir(exist_ok=True)

    subs = all_subjects() if args.subjects == 0 else range(1, args.subjects + 1)
    X, y, groups, ch_names = load_subject_data(subs, args.runs)
    log.info("%d epochs | %d subjects | %d ch x %d samples",
             len(X), len(np.unique(groups)), X.shape[1], X.shape[2])

    splits = list(GroupKFold(n_splits=args.n_splits).split(X, y, groups))
    json.dump({str(i): sorted(map(int, np.unique(groups[te])))
               for i, (_, te) in enumerate(splits, start=1)},
              open(RESULTS / "cv_splits.json", "w"), indent=2)

    rows = []

    X_smc = extract_psd_features(select_channels(X, ch_names, channels), sfreq=tfreq)
    X_all = extract_psd_features(X, sfreq=tfreq)
    rows += run_cv("PSD sensorimotor", psd_pipeline, X_smc, y, groups, splits, X_smc.shape[1])
    rows += run_cv("PSD all channels", psd_pipeline, X_all, y, groups, splits, X_all.shape[1])

    banks = [("FBCSP 4-40Hz (9 bands)", bank_4_40),
             ("FBCSP 4-32Hz (7 bands)", bank_4_32),
             ("FBCSP mu+beta (2 bands)", bank_mu_beta)]

    for name, bank in banks:
        Xb = apply_filter_bank(X, sfreq=tfreq, bank=bank)
        n_feat = min(args.k, len(bank) * 4)
        rows += run_cv(name, lambda b=bank: build_fbcsp(len(b), k=args.k),
                       Xb, y, groups, splits, n_feat)

        if bank is bank_4_40:
            for nc in (2, 8):
                rows += run_cv(f"FBCSP 4-40Hz, {nc} comp",
                               lambda b=bank, n=nc: build_fbcsp(len(b), n_components=n, k=args.k),
                               Xb, y, groups, splits, min(args.k, len(bank) * nc))
            for bi, tag in ((1, "mu"), (4, "beta")):
                plot_patterns(Xb, y, ch_names, bank, band_idx=bi,
                              path=RESULTS / f"csp_patterns_{tag}.png")
        del Xb

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(np.unique(groups)))

    print("\nfold-by-fold\n" + "-" * 92)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nsummary (mean over folds)\n" + "-" * 92)
    print(summarise(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nappended to {RESULTS / 'stage1_metrics.csv'} and {RESULTS / 'stage2_metrics.csv'}")


if __name__ == "__main__":
    main()

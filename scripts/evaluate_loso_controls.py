"""LOSO evaluation and the diagnostic controls that sit around it.

Each mode answers a question a skeptical reader asks about the headline number:

  loso               what is the spread across people, not the cohort mean
  leakage_test       how much would ignoring subject grouping have inflated it
  negative_control   does non-motor cortex score just as well
  executed_transfer  does a model trained on real movement survive on imagery
  csp_validation     does the covariance-domain CSP agree with mne.decoding.CSP
  subject_norm       does per-subject z-scoring rescue cross-subject CSP
  within_subject     the within-person ceiling LOSO is measured against
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.csp_fast import BandCSP, class_sums, filter_bank_covs
from src.data_loader import all_subjects, ex, im, load_subject_data, tfreq
from src.fbcsp import apply_filter_bank, bank_4_40, build_fbcsp
from src.features_psd import extract_psd_features, select_channels
from src.results import append_metrics, summarise

log = logging.getLogger("loso")
RESULTS = Path(__file__).resolve().parents[1] / "results"

CH_SETS = {
    "sensorimotor": ("C3", "Cz", "C4"),
    "occipital": ("O1", "Oz", "O2"),
    "frontal": ("Fp1", "Fp2"),
}


def linear():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def selective(k, seed=0):
    from functools import partial
    return Pipeline([("select", SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def fit_score(model, F, y, tr, te, F_te=None, y_te=None):
    """Score one split. F_te/y_te let train and test come from different data."""
    model.fit(F[tr], y[tr])
    Fte = F[te] if F_te is None else F_te[te]
    yte = y[te] if y_te is None else y_te[te]
    return {
        "train_acc": accuracy_score(y[tr], model.predict(F[tr])),
        "test_acc": accuracy_score(yte, model.predict(Fte)),
        "train_auc": roc_auc_score(y[tr], model.decision_function(F[tr])),
        "test_auc": roc_auc_score(yte, model.decision_function(Fte)),
    }


def row(method, fold, n_features, groups_te, n_te, scores):
    return {"method": method, "fold": fold, "n_features": n_features,
            "n_test_subjects": len(np.unique(groups_te)), "n_test_epochs": n_te, **scores}


def show(df, title):
    print(f"\n{title}\n" + "-" * 96)
    print(summarise(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))



def psd_sets(X, ch_names, which=("sensorimotor",), include_all=True):
    sets = {f"PSD {name}": extract_psd_features(select_channels(X, ch_names, CH_SETS[name]),
                                                sfreq=tfreq) for name in which}
    if include_all:
        sets["PSD all channels"] = extract_psd_features(X, sfreq=tfreq)
    return sets


def csp_loso_rows(C, y, groups, subjects, k, n_components, tag):
    """LOSO over subjects using total-minus-heldout covariance arithmetic."""
    tot_sums, tot_counts = class_sums(C, y)
    per = {s: class_sums(C[groups == s], y[groups == s]) for s in subjects}

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(subjects, 1):
        s_sums, s_counts = per[s]
        te = np.flatnonzero(groups == s)
        tr = np.flatnonzero(groups != s)

        csp = BandCSP(n_components).fit_from_sums(tot_sums - s_sums, tot_counts - s_counts)
        F = csp.transform(C)
        rows.append(row(tag, s, min(k, F.shape[1]), groups[te], len(te),
                        fit_score(selective(min(k, F.shape[1])), F, y, tr, te)))
        if i % 25 == 0:
            log.info("  %s %d/%d (%.0fs)", tag, i, len(subjects), time.perf_counter() - t0)
    log.info("%-28s %d folds in %.0fs", tag, len(rows), time.perf_counter() - t0)
    return rows


def linear_loso_rows(F, y, groups, subjects, tag):
    rows = []
    for s in subjects:
        te, tr = np.flatnonzero(groups == s), np.flatnonzero(groups != s)
        rows.append(row(tag, s, F.shape[1], groups[te], len(te), fit_score(linear(), F, y, tr, te)))
    log.info("%-28s %d folds", tag, len(rows))
    return rows



def mode_loso(args):
    subs = all_subjects()[: args.subjects] if args.subjects else all_subjects()
    X, y, g, ch = load_subject_data(subs, im)
    present = sorted(np.unique(g))
    log.info("%d epochs | %d subjects", len(X), len(present))

    rows = []
    for name, F in psd_sets(X, ch).items():
        rows += linear_loso_rows(F, y, g, present, name)

    C = filter_bank_covs(X, tfreq, bank_4_40)
    log.info("cov tensor %s (%.2f GB)", C.shape, C.nbytes / 1e9)
    rows += csp_loso_rows(C, y, g, present, args.k, 4, "FBCSP 4-40Hz LOSO")

    df = append_metrics(rows, cv="LOSO", n_subjects=len(present))
    show(df, "LOSO summary")

    dist = df[["method", "fold", "n_test_epochs", "test_acc", "test_auc"]].rename(
        columns={"fold": "subject"})
    dist.to_csv(RESULTS / "loso_distributions.csv", index=False)
    print(f"\nwrote {RESULTS / 'loso_distributions.csv'}")

    print("\nper-subject distribution\n" + "-" * 96)
    for m, sub in dist.groupby("method", sort=False):
        a = sub.test_acc.values
        q = np.percentile(a, [0, 25, 50, 75, 100])
        print(f"{m:<24} min={q[0]:.3f} q1={q[1]:.3f} med={q[2]:.3f} q3={q[3]:.3f} "
              f"max={q[4]:.3f} | <=chance: {100*np.mean(a <= 0.5):.0f}% | >0.6: {100*np.mean(a > 0.6):.0f}%")


def mode_leakage_test(args):
    subs = all_subjects()[: args.subjects] if args.subjects else all_subjects()
    X, y, g, ch = load_subject_data(subs, im)

    feats = psd_sets(X, ch)
    C = filter_bank_covs(X, tfreq, bank_4_40)

    grouped = list(GroupKFold(args.n_splits).split(X, y, g))
    naive = list(StratifiedKFold(args.n_splits, shuffle=True, random_state=0).split(X, y))

    dfs = []
    for scheme, splits in (("GroupKFold", grouped), ("StratifiedKFold", naive)):
        rows = []
        for name, F in feats.items():
            for fold, (tr, te) in enumerate(splits, 1):
                rows.append(row(name, fold, F.shape[1], g[te], len(te),
                                fit_score(linear(), F, y, tr, te)))
        for fold, (tr, te) in enumerate(splits, 1):
            s_, n_ = class_sums(C[tr], y[tr])
            F = BandCSP(4).fit_from_sums(s_, n_).transform(C)
            k = min(args.k, F.shape[1])
            rows.append(row("FBCSP 4-40Hz", fold, k, g[te], len(te),
                            fit_score(selective(k), F, y, tr, te)))
        dfs.append(append_metrics(rows, cv=f"{scheme}{args.n_splits}",
                                  n_subjects=len(np.unique(g))))

    df = pd.concat(dfs)
    show(df, "leakage stress test -- GroupKFold vs subject-blind StratifiedKFold")

    acc = summarise(df).pivot(index="method", columns="cv", values="test_acc")
    grouped_cv, blind_cv = f"GroupKFold{args.n_splits}", f"StratifiedKFold{args.n_splits}"
    print("\ninflation from ignoring subject grouping\n" + "-" * 96)
    for method, r in acc.iterrows():
        print(f"{method:<24} grouped={r[grouped_cv]:.4f}  blind={r[blind_cv]:.4f}  "
              f"inflation={r[blind_cv] - r[grouped_cv]:+.4f}")


def mode_negative_control(args):
    subs = all_subjects()[: args.subjects] if args.subjects else all_subjects()
    X, y, g, ch = load_subject_data(subs, im)

    missing = {n: [c for c in cs if c.upper() not in {v.upper() for v in ch}]
               for n, cs in CH_SETS.items()}
    for n, m in missing.items():
        if m:
            raise SystemExit(f"channel set {n} missing {m}")

    rows = []
    splits = list(GroupKFold(args.n_splits).split(X, y, g))
    for name in CH_SETS:
        F = extract_psd_features(select_channels(X, ch, CH_SETS[name]), sfreq=tfreq)
        for fold, (tr, te) in enumerate(splits, 1):
            rows.append(row(f"PSD {name}", fold, F.shape[1], g[te], len(te),
                            fit_score(linear(), F, y, tr, te)))

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(np.unique(g)))
    show(df, "negative-control channels (all PSD mu+beta, GroupKFold)")
    print("\nOccipital/frontal scoring like sensorimotor would mean the signal is "
          "not motor cortex.")


def mode_executed_transfer(args):
    subs = all_subjects()[: args.subjects] if args.subjects else all_subjects()
    Xe, ye, ge, ch = load_subject_data(subs, ex)
    Xi, yi, gi, _ = load_subject_data(subs, im)
    log.info("executed %s | imagined %s", Xe.shape, Xi.shape)

    common = sorted(set(np.unique(ge)) & set(np.unique(gi)))
    rows = []
    for fold, (tr_s, te_s) in enumerate(
            GroupKFold(args.n_splits).split(common, groups=common), 1):
        train_subj = {common[i] for i in tr_s}
        test_subj = {common[i] for i in te_s}
        tr = np.flatnonzero(np.isin(ge, list(train_subj)))
        te_i = np.flatnonzero(np.isin(gi, list(test_subj)))
        te_e = np.flatnonzero(np.isin(ge, list(test_subj)))

        Fe = extract_psd_features(select_channels(Xe, ch, CH_SETS["sensorimotor"]), sfreq=tfreq)
        Fi = extract_psd_features(select_channels(Xi, ch, CH_SETS["sensorimotor"]), sfreq=tfreq)

        m = linear().fit(Fe[tr], ye[tr])
        for tag, F, yy, idx in (("PSD exec->exec", Fe, ye, te_e), ("PSD exec->imag", Fi, yi, te_i)):
            rows.append({"method": tag, "fold": fold, "n_features": Fe.shape[1],
                         "n_test_subjects": len(test_subj), "n_test_epochs": len(idx),
                         "train_acc": accuracy_score(ye[tr], m.predict(Fe[tr])),
                         "test_acc": accuracy_score(yy[idx], m.predict(F[idx])),
                         "train_auc": roc_auc_score(ye[tr], m.decision_function(Fe[tr])),
                         "test_auc": roc_auc_score(yy[idx], m.decision_function(F[idx]))})

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(common))
    show(df, "executed -> imagined transfer (trained on runs 3/7/11)")


def mode_csp_validation(args):
    """Covariance-domain CSP against mne.decoding.CSP on identical splits.

    Both get all 36 log-variance features and the same linear classifier, so
    any gap is the CSP estimator itself (trace-norm + ridge vs Ledoit-Wolf).
    Defaults to 20 subjects: the mne path needs the whole filtered tensor in
    memory, about 1 GB per 20 subjects.
    """
    X, y, g, ch = load_subject_data(all_subjects()[: args.subjects or 20], im)
    splits = list(GroupKFold(args.n_splits).split(X, y, g))
    n_feat = len(bank_4_40) * 4
    rows = []

    C = filter_bank_covs(X, tfreq, bank_4_40)
    for fold, (tr, te) in enumerate(splits, 1):
        s_, n_ = class_sums(C[tr], y[tr])
        F = BandCSP(4).fit_from_sums(s_, n_).transform(C)
        rows.append(row("CSP covariance-domain", fold, n_feat, g[te], len(te),
                        fit_score(linear(), F, y, tr, te)))
    del C

    Xb = apply_filter_bank(X, sfreq=tfreq, bank=bank_4_40)
    for fold, (tr, te) in enumerate(splits, 1):
        rows.append(row("CSP mne ledoit-wolf", fold, n_feat, g[te], len(te),
                        fit_score(build_fbcsp(len(bank_4_40), k=n_feat), Xb, y, tr, te)))

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(np.unique(g)))
    show(df, "CSP implementation check -- same splits, all 36 features, no selection")


def mode_subject_norm(args):
    """Does removing each subject's per-channel gain rescue cross-subject CSP?

    Epoch covariances are already trace-normalised, so global amplitude is
    gone either way; what z-scoring additionally removes is the *relative*
    gain between channels. If that were the transfer problem, this would help.
    """
    subs = all_subjects()[: args.subjects] if args.subjects else all_subjects()
    X, y, g, ch = load_subject_data(subs, im)
    splits = list(GroupKFold(args.n_splits).split(X, y, g))

    Xn = X.copy()
    for s in np.unique(g):
        m = g == s
        Xn[m] = (Xn[m] - Xn[m].mean(axis=(0, 2), keepdims=True)) / Xn[m].std(axis=(0, 2), keepdims=True)

    rows = []
    for tag, D in (("FBCSP 4-40Hz raw", X), ("FBCSP 4-40Hz subject-zscored", Xn)):
        C = filter_bank_covs(D, tfreq, bank_4_40)
        for fold, (tr, te) in enumerate(splits, 1):
            s_, n_ = class_sums(C[tr], y[tr])
            F = BandCSP(4).fit_from_sums(s_, n_).transform(C)
            k = min(args.k, F.shape[1])
            rows.append(row(tag, fold, k, g[te], len(te), fit_score(selective(k), F, y, tr, te)))
        del C

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(np.unique(g)))
    show(df, "per-subject z-scoring before the filter bank")


def mode_within_subject(args):
    """Train and test inside each subject: the ceiling LOSO is measured against.

    45 trials per subject leaves ~36 training epochs per fold, so CSP is
    estimating 64x64 covariances from almost nothing and every per-subject
    number is noisy. Read the cohort mean, not the rows. One row per subject,
    metrics averaged over its folds so it lines up with loso_distributions.
    """
    subs = all_subjects()[: args.subjects] if args.subjects else all_subjects()
    X, y, g, ch = load_subject_data(subs, im)
    present = sorted(np.unique(g))

    F_psd = extract_psd_features(select_channels(X, ch, CH_SETS["sensorimotor"]), sfreq=tfreq)
    C = filter_bank_covs(X, tfreq, bank_4_40)
    k = min(args.k, C.shape[1] * 4)

    rows = []
    for s in present:
        idx = np.flatnonzero(g == s)
        Fp, Cs, ys = F_psd[idx], C[idx], y[idx]
        per = {"PSD sensorimotor": [], "FBCSP 4-40Hz": []}
        for tr, te in StratifiedKFold(args.n_splits, shuffle=True, random_state=0).split(Fp, ys):
            per["PSD sensorimotor"].append(fit_score(linear(), Fp, ys, tr, te))
            s_, n_ = class_sums(Cs[tr], ys[tr])
            Fc = BandCSP(4).fit_from_sums(s_, n_).transform(Cs)
            per["FBCSP 4-40Hz"].append(fit_score(selective(k), Fc, ys, tr, te))
        for tag, scores in per.items():
            mean = {m: float(np.mean([d[m] for d in scores])) for m in scores[0]}
            rows.append(row(tag, s, F_psd.shape[1] if tag.startswith("PSD") else k,
                            g[idx], len(idx), mean))

    df = append_metrics(rows, cv=f"within-subject StratifiedKFold{args.n_splits}",
                        n_subjects=len(present))
    show(df, "within-subject ceiling (one row per subject, folds averaged)")


MODES = {"loso": mode_loso, "leakage_test": mode_leakage_test,
         "negative_control": mode_negative_control, "executed_transfer": mode_executed_transfer,
         "csp_validation": mode_csp_validation, "subject_norm": mode_subject_norm,
         "within_subject": mode_within_subject}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--subjects", type=int, default=0, help="0 = all 106 clean subjects")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mne.set_log_level("ERROR")
    RESULTS.mkdir(exist_ok=True)
    MODES[args.mode](args)


if __name__ == "__main__":
    main()

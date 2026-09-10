import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
from joblib import Parallel, delayed
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.cache import cached_load
from src.data_loader import all_subjects, ex, im, tfreq
from src.features_psd import select_channels
from src.results import RESULTS, append_metrics, summarise
from src.riemannian import bandpass, broad_band, clf_head, epoch_covs, fb_bands, recenter
from scripts.evaluate_loso_controls import CH_SETS, fit_score, row, show

log = logging.getLogger("stage3")
STAGE = 3

TS_RAW = "Riemannian TS 1-35Hz"
TS_BB = "Riemannian TS 7-30Hz"
TS_RC = "Riemannian TS 7-30Hz recentered"
TS_FB = "FB-Riemannian 3-band recentered"


def cov_variants(X, g, ch_names=None, chans=None, with_fb=False):
    if chans is not None:
        X = select_channels(X, ch_names, chans)
    lo, hi = broad_band
    Xb = bandpass(X, lo, hi, sfreq=tfreq)
    C_bb = epoch_covs(Xb)
    out = {
        TS_RAW: (epoch_covs(X), False),
        TS_BB: (C_bb, False),
        TS_RC: (recenter(C_bb, g), True),
    }
    if with_fb:
        parts = [recenter(epoch_covs(bandpass(X, a, b, sfreq=tfreq)), g) for a, b in fb_bands]
        out[TS_FB] = (np.stack(parts, axis=1), True)
    return out


def fixed_features(C):
    if C.ndim == 4:
        eye = np.eye(C.shape[-1])
        return np.hstack([tangent_space(C[:, b], eye) for b in range(C.shape[1])])
    return tangent_space(C, np.eye(C.shape[-1]))


def fold_features(C, tr, init):
    ref = mean_riemann(C[tr], init=init, tol=1e-5)
    return tangent_space(C, ref)


def score_rows(name, C, recentered, y, g, splits, fold_ids, n_jobs):
    t0 = time.perf_counter()

    def one(F, tr, te, fid):
        r = row(name, fid, F.shape[1], g[te], len(te), fit_score(clf_head(), F, y, tr, te))
        r["stage"] = STAGE
        return r

    if recentered:
        F = fixed_features(C)
        rows = Parallel(n_jobs=n_jobs)(delayed(one)(F, tr, te, fid)
                                       for (tr, te), fid in zip(splits, fold_ids))
    else:
        init = mean_riemann(C)

        def one_fold(tr, te, fid):
            return one(fold_features(C, tr, init), tr, te, fid)

        rows = Parallel(n_jobs=n_jobs)(delayed(one_fold)(tr, te, fid)
                                       for (tr, te), fid in zip(splits, fold_ids))

    log.info("%-36s %d folds in %.0fs  test_acc=%.4f", name, len(rows),
             time.perf_counter() - t0, np.mean([r["test_acc"] for r in rows]))
    return rows


def subjects_arg(args):
    return all_subjects()[: args.subjects] if args.subjects else all_subjects()


def mode_loso(args):
    subs = subjects_arg(args)
    X, y, g, ch = cached_load(subs, im)
    present = sorted(np.unique(g))
    splits = [(np.flatnonzero(g != s), np.flatnonzero(g == s)) for s in present]

    rows = []
    for name, (C, rc) in cov_variants(X, g).items():
        rows += score_rows(name, C, rc, y, g, splits, present, args.n_jobs)

    df = append_metrics(rows, cv="LOSO", n_subjects=len(present))
    show(df, "Stage 3 LOSO")

    dist = df[["method", "fold", "n_test_epochs", "test_acc", "test_auc"]].rename(columns={"fold": "subject"})
    dist_path = RESULTS / "loso_distributions.csv"
    dist.to_csv(dist_path, mode="a", header=not dist_path.exists(), index=False)

    full = pd.read_csv(dist_path)
    keep = {"PSD sensorimotor": "psd_acc", "FBCSP 4-40Hz LOSO": "fbcsp_acc",
            TS_BB: "riemannian_acc", TS_RC: "riemannian_recentered_acc"}
    wide = (full[full.method.isin(keep)].drop_duplicates(["method", "subject"], keep="last")
            .pivot(index="subject", columns="method", values="test_acc")
            .rename(columns=keep))
    wide.to_csv(RESULTS / "loso_distributions_wide.csv")

    print("\nper-subject distribution\n" + "-" * 96)
    for m, sub in dist.groupby("method", sort=False):
        a = sub.test_acc.values
        q = np.percentile(a, [0, 25, 50, 75, 100])
        print(f"{m:<36} min={q[0]:.3f} q1={q[1]:.3f} med={q[2]:.3f} q3={q[3]:.3f} "
              f"max={q[4]:.3f} | <=chance: {100*np.mean(a <= 0.5):.0f}% | >0.6: {100*np.mean(a > 0.6):.0f}%")


def mode_group_kfold(args):
    subs = subjects_arg(args)
    X, y, g, ch = cached_load(subs, im)
    splits = list(GroupKFold(args.n_splits).split(X, y, g))
    ids = list(range(1, len(splits) + 1))

    rows = []
    for name, (C, rc) in cov_variants(X, g, with_fb=True).items():
        rows += score_rows(name, C, rc, y, g, splits, ids, args.n_jobs)

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(np.unique(g)))
    show(df, "Stage 3 GroupKFold")


def mode_leakage_test(args):
    subs = subjects_arg(args)
    X, y, g, ch = cached_load(subs, im)
    grouped = list(GroupKFold(args.n_splits).split(X, y, g))
    naive = list(StratifiedKFold(args.n_splits, shuffle=True, random_state=0).split(X, y))
    ids = list(range(1, args.n_splits + 1))

    variants = {k: v for k, v in cov_variants(X, g).items() if k in (TS_BB, TS_RC)}
    dfs = []
    for scheme, splits in (("GroupKFold", grouped), ("StratifiedKFold", naive)):
        rows = []
        for name, (C, rc) in variants.items():
            rows += score_rows(name, C, rc, y, g, splits, ids, args.n_jobs)
        dfs.append(append_metrics(rows, cv=f"{scheme}{args.n_splits}", n_subjects=len(np.unique(g))))

    df = pd.concat(dfs)
    show(df, "Stage 3 leakage stress test")
    acc = summarise(df).pivot(index="method", columns="cv", values="test_acc")
    gk, sk = f"GroupKFold{args.n_splits}", f"StratifiedKFold{args.n_splits}"
    print("\ninflation from ignoring subject grouping\n" + "-" * 96)
    for m, r in acc.iterrows():
        print(f"{m:<36} grouped={r[gk]:.4f}  blind={r[sk]:.4f}  inflation={r[sk]-r[gk]:+.4f}")


def mode_negative_control(args):
    subs = subjects_arg(args)
    X, y, g, ch = cached_load(subs, im)
    splits = list(GroupKFold(args.n_splits).split(X, y, g))
    ids = list(range(1, len(splits) + 1))

    rows = []
    for region, chans in CH_SETS.items():
        variants = cov_variants(X, g, ch, chans)
        for base in (TS_BB, TS_RC):
            C, rc = variants[base]
            tag = base.replace("Riemannian TS", f"Riemannian {region}")
            rows += score_rows(tag, C, rc, y, g, splits, ids, args.n_jobs)

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(np.unique(g)))
    show(df, "Stage 3 negative-control channels")


def mode_executed_transfer(args):
    subs = subjects_arg(args)
    Xe, ye, ge, ch = cached_load(subs, ex)
    Xi, yi, gi, _ = cached_load(subs, im)
    common = sorted(set(np.unique(ge)) & set(np.unique(gi)))

    lo, hi = broad_band
    Ce = epoch_covs(bandpass(Xe, lo, hi, sfreq=tfreq))
    Ci = epoch_covs(bandpass(Xi, lo, hi, sfreq=tfreq))
    sets = {TS_BB: (Ce, Ci, False), TS_RC: (recenter(Ce, ge), recenter(Ci, gi), True)}

    rows = []
    for fold, (tr_s, te_s) in enumerate(GroupKFold(args.n_splits).split(common, groups=common), 1):
        train_subj = [common[i] for i in tr_s]
        test_subj = [common[i] for i in te_s]
        tr = np.flatnonzero(np.isin(ge, train_subj))
        te_e = np.flatnonzero(np.isin(ge, test_subj))
        te_i = np.flatnonzero(np.isin(gi, test_subj))

        for base, (Ce_, Ci_, rc) in sets.items():
            if rc:
                Fe, Fi = fixed_features(Ce_), fixed_features(Ci_)
            else:
                ref = mean_riemann(Ce_[tr], tol=1e-5)
                Fe, Fi = tangent_space(Ce_, ref), tangent_space(Ci_, ref)
            m = clf_head().fit(Fe[tr], ye[tr])
            for tag, F, yy, idx in ((f"{base} exec->exec", Fe, ye, te_e), (f"{base} exec->imag", Fi, yi, te_i)):
                r = row(tag, fold, Fe.shape[1], np.array(test_subj), len(idx), {
                    "train_acc": (m.predict(Fe[tr]) == ye[tr]).mean(),
                    "test_acc": (m.predict(F[idx]) == yy[idx]).mean(),
                    "train_auc": __import__("sklearn.metrics").metrics.roc_auc_score(ye[tr], m.decision_function(Fe[tr])),
                    "test_auc": __import__("sklearn.metrics").metrics.roc_auc_score(yy[idx], m.decision_function(F[idx])),
                })
                r["stage"] = STAGE
                rows.append(r)

    df = append_metrics(rows, cv=f"GroupKFold{args.n_splits}", n_subjects=len(common))
    show(df, "Stage 3 executed -> imagined transfer")


def mode_c_sweep(args):
    subs = subjects_arg(args)
    X, y, g, ch = cached_load(subs, im)
    lo, hi = broad_band
    C_rc = recenter(epoch_covs(bandpass(X, lo, hi, sfreq=tfreq)), g)
    F = fixed_features(C_rc)
    present = sorted(np.unique(g))
    loso = [(np.flatnonzero(g != s), np.flatnonzero(g == s)) for s in present]
    gk = list(GroupKFold(args.n_splits).split(X, y, g))

    rows = []
    for c in (0.001, 0.01, 0.1, 1.0):
        name = f"{TS_RC} C={c:g}"
        for cv_name, splits, ids in (("LOSO", loso, present), (f"GroupKFold{args.n_splits}", gk, list(range(1, len(gk) + 1)))):
            def one(tr, te, fid):
                r = row(name, fid, F.shape[1], g[te], len(te), fit_score(clf_head(C=c), F, y, tr, te))
                r["stage"] = STAGE
                return r
            t0 = time.perf_counter()
            part = Parallel(n_jobs=args.n_jobs)(delayed(one)(tr, te, fid) for (tr, te), fid in zip(splits, ids))
            log.info("%-40s %-12s %d folds in %.0fs  test_acc=%.4f  train_acc=%.4f", name, cv_name, len(part),
                     time.perf_counter() - t0, np.mean([r["test_acc"] for r in part]), np.mean([r["train_acc"] for r in part]))
            append_metrics(part, cv=cv_name, n_subjects=len(present))
            rows += [dict(r, cv=cv_name) for r in part]

    df = pd.DataFrame(rows)
    df["n_subjects"] = len(present)
    show(df, "Stage 3 regularization sweep (recentered tangent space)")


MODES = {"loso": mode_loso, "group_kfold": mode_group_kfold, "leakage_test": mode_leakage_test,
         "negative_control": mode_negative_control, "executed_transfer": mode_executed_transfer,
         "c_sweep": mode_c_sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mne.set_log_level("ERROR")
    RESULTS.mkdir(exist_ok=True)
    MODES[args.mode](args)


if __name__ == "__main__":
    main()

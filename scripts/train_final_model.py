import argparse
import logging
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import pyriemann
import sklearn

from src.cache import cached_load
from src.data_loader import all_subjects, im, lower, tfreq, tmax, tmin, upper
from src.riemannian import RecenteredTangentClassifier, broad_band

log = logging.getLogger("train")
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "riemannian_model.joblib"


def fit_final(subjects=None, C=1.0, runs=im):
    subs = subjects or all_subjects()
    X, y, groups, ch_names = cached_load(subs, runs)
    log.info("%d epochs | %d subjects | %d ch x %d samples", len(X), len(np.unique(groups)), X.shape[1], X.shape[2])

    t0 = time.perf_counter()
    model = RecenteredTangentClassifier(band=broad_band, sfreq=tfreq, C=C)
    model.fit(X, y, groups)
    model.ch_names = list(ch_names)
    model.meta = {
        "runs": list(runs),
        "subjects": sorted(int(s) for s in np.unique(groups)),
        "n_epochs": int(len(X)),
        "raw_filter_hz": (lower, upper),
        "epoch_window_s": (tmin, tmax),
        "cov_band_hz": broad_band,
        "cov_estimator": "lwf",
        "recentering": "per-subject riemannian mean, unsupervised",
        "classifier": f"StandardScaler + LogisticRegression(C={C})",
        "labels": {0: "Left Fist (T1)", 1: "Right Fist (T2)"},
        "versions": {"python": platform.python_version(), "mne": mne.__version__,
                     "pyriemann": pyriemann.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__},
    }
    log.info("fit in %.0fs, train acc %.4f", time.perf_counter() - t0,
             (model.predict(X, groups) == y).mean())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mne.set_log_level("ERROR")

    model = fit_final(C=args.C)
    args.out.parent.mkdir(exist_ok=True)
    joblib.dump(model, args.out, compress=3)
    print(f"\nsaved {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    for k, v in model.meta.items():
        if k != "subjects":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

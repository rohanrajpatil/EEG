import hashlib
import logging
from pathlib import Path

import numpy as np

from src.data_loader import load_subject_data

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _key(subjects, runs):
    subs = ",".join(str(s) for s in subjects)
    rs = ",".join(str(r) for r in runs)
    h = hashlib.md5(f"{subs}|{rs}".encode()).hexdigest()[:10]
    return f"epochs_r{'-'.join(str(r) for r in runs)}_n{len(list(subjects))}_{h}.npz"


def cached_load(subjects, runs):
    subjects = list(subjects)
    runs = list(runs)
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / _key(subjects, runs)

    if path.exists():
        z = np.load(path, allow_pickle=False)
        log.info("cache hit %s", path.name)
        return z["X"], z["y"], z["groups"], list(z["ch_names"])

    X, y, groups, ch_names = load_subject_data(subjects, runs)
    np.savez(path, X=X.astype(np.float32), y=y, groups=groups,
             ch_names=np.array(ch_names))
    log.info("cache write %s (%.2f GB)", path.name, path.stat().st_size / 1e9)
    return X.astype(np.float32), y, groups, ch_names

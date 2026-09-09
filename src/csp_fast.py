"""CSP expressed over epoch covariances rather than raw signal.

The point is leave-one-subject-out. A class covariance is a sum over epochs, so
the training covariance for a fold is the cohort total minus the held-out
subject -- a subtraction instead of a refit. That turns 106 CSP folds from hours
into seconds, at the cost of writing the eigendecomposition out by hand instead
of calling mne.decoding.CSP.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt


def epoch_covs(X):
    """(n_epochs, n_ch, n_times) -> (n_epochs, n_ch, n_ch), normalised by trace.

    Trace normalisation makes each epoch's covariance scale-free, so a subject
    with large absolute amplitude cannot dominate the pooled estimate purely
    through gain.
    """
    C = X @ X.transpose(0, 2, 1) / X.shape[-1]
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


def class_sums(C, y, n_classes=2):
    """Per-class covariance sums and counts -- the quantities that are additive.

    C: (n_epochs, n_bands, n_ch, n_ch) -> sums (n_bands, n_classes, n_ch, n_ch).
    """
    sums = np.stack([C[y == k].sum(axis=0, dtype=np.float64)
                     for k in range(n_classes)], axis=1)
    counts = np.array([(y == k).sum() for k in range(n_classes)], dtype=np.int64)
    return sums, counts


def _shrink(C, alpha):
    # Ridge toward a scaled identity. With 64 channels the pooled covariance is
    # workable but its small eigenvalues are badly estimated, and CSP divides by
    # them -- shrinkage is what stops the filters chasing those directions.
    return (1 - alpha) * C + alpha * np.trace(C) / C.shape[0] * np.eye(C.shape[0])


def csp_filters(C0, C1, n_components=4, alpha=0.1):
    """Generalised eigendecomposition of C0 against the composite covariance.

    Components are taken in pairs from opposite ends of the spectrum: the
    extremes maximise the variance ratio between classes, while the middle of
    the spectrum is where the two classes look alike and carries no contrast.
    """
    C0, C1 = _shrink(C0, alpha), _shrink(C1, alpha)
    w, V = eigh(C0, C0 + C1)          # ascending eigenvalues
    order = np.argsort(w)

    picks = []
    lo, hi = 0, len(w) - 1
    while len(picks) < n_components:
        picks.append(order[hi]); hi -= 1
        if len(picks) < n_components:
            picks.append(order[lo]); lo += 1
    return V[:, picks].T              # (n_components, n_ch)


def log_var_features(C, W):
    """Log-variance of each CSP projection, read straight off epoch covariances.

    var_k = w_k' C w_k, so no time-domain projection is needed. Normalising by
    the total across components before the log is the standard CSP feature: it
    removes what is left of per-epoch power and keeps the ratio between them.
    """
    v = np.einsum("kc,ecd,kd->ek", W, C, W)
    return np.log(v / v.sum(axis=1, keepdims=True) + 1e-20)


class BandCSP:
    """Fits one CSP per sub-band from precomputed per-class covariance sums."""

    def __init__(self, n_components=4, alpha=0.1):
        self.n_components = n_components
        self.alpha = alpha

    def fit_from_sums(self, sums, counts):
        """sums: (n_bands, 2, n_ch, n_ch), counts: (2,) -- already train-only."""
        self.filters_ = [
            csp_filters(sums[b, 0] / counts[0], sums[b, 1] / counts[1],
                        self.n_components, self.alpha)
            for b in range(len(sums))
        ]
        return self

    def transform(self, C):
        """C: (n_epochs, n_bands, n_ch, n_ch) -> (n_epochs, n_bands * n_components)."""
        return np.hstack([log_var_features(C[:, b], W)
                          for b, W in enumerate(self.filters_)])


def filter_bank_covs(X, sfreq, bank, order=4):
    """Per-band epoch covariances, one band at a time.

    Materialising the whole filtered tensor for 106 subjects would cost ~5 GB;
    the covariances are 8x smaller and are all the CSP path actually needs, so
    each band's filtered signal is discarded as soon as it is reduced.
    """
    out = np.empty((len(X), len(bank), X.shape[1], X.shape[1]), dtype=np.float32)
    for i, (lo, hi) in enumerate(bank):
        sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        out[:, i] = epoch_covs(sosfiltfilt(sos, X, axis=-1))
    return out

from functools import partial

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_bank(fmin=4.0, fmax=40.0, width=4.0, step=4.0):
    lo_freqs = np.arange(fmin, fmax - width + 1e-9, step)
    bands_list = []
    for lo in lo_freqs:
        hi = lo + width
        bands_list.append((float(lo), float(hi)))
    return bands_list


bank_4_40 = make_bank(4.0, 40.0, 4.0)
bank_4_32 = make_bank(4.0, 32.0, 4.0)
bank_mu_beta = [(8.0, 12.0), (13.0, 30.0)]


def apply_filter_bank(X, sfreq=160.0, bank=bank_4_40, order=4):
    n_epochs = X.shape[0]
    n_bands = len(bank)
    n_channels = X.shape[1]
    n_times = X.shape[2]

    filtered = np.empty((n_epochs, n_bands, n_channels, n_times), dtype=np.float32)

    for band_idx, (lo_freq, hi_freq) in enumerate(bank):
        sos_coeffs = butter(order, [lo_freq, hi_freq], btype="bandpass", fs=sfreq, output="sos")
        band_data = sosfiltfilt(sos_coeffs, X, axis=-1)
        filtered[:, band_idx, :, :] = band_data.astype(np.float32)

    return filtered


class FilterBankCSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_bands, n_components=4, reg="ledoit_wolf"):
        self.n_bands = n_bands
        self.n_components = n_components
        self.reg = reg

    def _make_csp(self):
        csp_obj = CSP(n_components=self.n_components, reg=self.reg,
                      log=True, norm_trace=False)
        return csp_obj

    def fit(self, X, y):
        self.csps_ = []
        for b in range(self.n_bands):
            csp_fitted = self._make_csp()
            csp_fitted.fit(X[:, b], y)
            self.csps_.append(csp_fitted)
        return self

    def transform(self, X):
        out_feats = []
        for band_idx, csp_obj in enumerate(self.csps_):
            feat_b = csp_obj.transform(X[:, band_idx])
            out_feats.append(feat_b)
        all_feats = np.hstack(out_feats)
        return all_feats


def build_fbcsp(n_bands, n_components=4, k=12, C=1.0, random_state=0):
    total_features = n_bands * n_components
    k_actual = min(k, total_features)

    mi_scorer = partial(mutual_info_classif, random_state=random_state)

    steps = [
        ("fbcsp", FilterBankCSP(n_bands=n_bands, n_components=n_components)),
        ("select", SelectKBest(mi_scorer, k=k_actual)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ]

    return Pipeline(steps)

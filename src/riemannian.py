import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

broad_band = (7.0, 30.0)
fb_bands = [(7.0, 13.0), (13.0, 20.0), (20.0, 30.0)]


def bandpass(X, lo, hi, sfreq=160.0, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def build_riemannian_pipeline(C=1.0):
    steps = [
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace(metric="riemann")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ]
    return Pipeline(steps)


def epoch_covs(X):
    return Covariances(estimator="lwf").transform(X.astype(np.float64))


def recenter(covs, groups):
    out = np.empty_like(covs)
    for s in np.unique(groups):
        m = groups == s
        ref = mean_riemann(covs[m])
        w = invsqrtm(ref)
        out[m] = w @ covs[m] @ w
    return out


class TangentFeatures:

    def __init__(self, init=None):
        self.init = init

    def fit(self, covs):
        self.ref_ = mean_riemann(covs, init=self.init)
        return self

    def transform(self, covs):
        return tangent_space(covs, self.ref_)


def clf_head(C=1.0):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=1000))])

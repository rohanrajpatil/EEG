from __future__ import annotations

from typing import Sequence

import numpy as np
from mne.time_frequency import psd_array_welch

bands= { "mu": (8.0, 12.0), "beta": (13.0, 30.0),}

channels =("C3", "Cz", "C4")


def extract_psd_features(epochs_data,sfreq = 160.0,bands = bands,n_per_seg = 256,):
    n = epochs_data.shape[-1]

    n_per_seg = min(n_per_seg, n)

    psds, freqs = psd_array_welch(
        epochs_data,
        sfreq=sfreq,
        fmin=1.0,
        fmax=40.0,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_per_seg // 2,
        average="mean",
        verbose="ERROR",
    )

    feats = []
    for fmin, fmax in bands.values():
        mask = (freqs >= fmin) & (freqs <= fmax)
        feats.append(np.log(psds[:, :, mask].mean(axis=-1) + 1e-20))

    return np.stack(feats, axis=0).transpose(1, 2, 0).reshape(len(epochs_data), -1)


def select_channels(epochs_data,ch_names,wanted,):
    lookup = {name.upper(): i for i, name in enumerate(ch_names)}
    return epochs_data[:, [lookup[c.upper()] for c in wanted], :]

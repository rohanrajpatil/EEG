import numpy as np
from mne.time_frequency import psd_array_welch

bands = {
    "mu": (8.0, 12.0),
    "beta": (13.0, 30.0),
}

channels = ("C3", "Cz", "C4")

DEFAULT_SEG = 256


def extract_psd_features(epochs_data, sfreq=160.0, bands=bands, n_per_seg=256):
    n_times = epochs_data.shape[-1]
    seg_len = min(n_per_seg, n_times)

    psds, freqs = psd_array_welch(
        epochs_data,
        sfreq=sfreq,
        fmin=1.0,
        fmax=40.0,
        n_fft=seg_len,
        n_per_seg=seg_len,
        n_overlap=seg_len // 2,
        average="mean",
        verbose="ERROR",
    )

    feature_list = []
    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psds[:, :, freq_mask].mean(axis=-1)
        log_band = np.log(band_power + 1e-20)
        feature_list.append(log_band)

    stacked = np.stack(feature_list, axis=0)
    transposed = stacked.transpose(1, 2, 0)
    flattened = transposed.reshape(len(epochs_data), -1)

    return flattened


def select_channels(epochs_data, ch_names, wanted):
    ch_lookup = {}
    for i, name in enumerate(ch_names):
        ch_lookup[name.upper()] = i

    indices = [ch_lookup[c.upper()] for c in wanted]
    result = epochs_data[:, indices, :]

    return result

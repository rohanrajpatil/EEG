from __future__ import annotations

import logging
import mne
import numpy as np
from mne.datasets import eegbci

log = logging.getLogger(__name__)



im = (4, 8, 12)
ex = (3, 7, 11)

id = {"T1": 1, "T2": 2}

lower, upper= 1.0, 35.0
tmin, tmax = 0.5, 3.5
tfreq = 160.0

excluded = {88, 92, 100}


def all_subjects():
    return [s for s in range(1, 110) if s not in excluded]

def read_runs(subject, runs):
    paths = eegbci.load_data(subject, list(runs), update_path=True, verbose="error")
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="error") for p in paths]
    raw = mne.concatenate_raws(raws, verbose="error")

    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose="error")


    if not np.isclose(raw.info["sfreq"], tfreq):
        raw.resample(tfreq, verbose="error")

    raw.filter(lower, upper, fir_design="firwin", skip_by_annotation="edge", verbose="error")
    return raw


def split(raw):
    events, x= mne.events_from_annotations(raw, event_id=id, verbose="error")

    return mne.Epochs(
        raw,
        events,
        event_id=id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )


def load_subject_data( ids, runs):
    rvrse = {1: 0 , 2: 1}
    epochs_data = []
    labels = []
    groups= []
    ch_names = None

    for id in ids:
        if id in excluded:
            log.warning("subject %s excluded (non-standard recording)", id)
            continue
        try:
            raw = read_runs(id, runs)
            epochs = split(raw)

            if len(epochs) == 0:
                log.warning("no usable epochs, skipping %s", id)
                continue

            if ch_names is None:
                ch_names = epochs.ch_names
            elif epochs.ch_names != ch_names:
                log.warning("mismatched channel set, skipping %s", id)
                continue

            data = epochs.get_data(copy=False)
            y = np.array([rvrse[code] for code in epochs.events[:, 2]], dtype=np.int64)

            epochs_data.append(data)
            labels.append(y)
            groups.append(np.full(len(y), id, dtype=np.int64))

        except Exception:
            log.exception("failed to load subject %s", id)
            continue

    if not epochs_data:
        raise RuntimeError("no subjects loaded successfully")

    n = min(arr.shape[-1] for arr in epochs_data)
    x = np.concatenate([arr[..., :n] for arr in epochs_data], axis=0)

    return x, np.concatenate(labels), np.concatenate(groups), ch_names

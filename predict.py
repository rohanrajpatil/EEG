import argparse
import re
import sys
from pathlib import Path

import joblib
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import id as event_id, lower, tfreq, tmax, tmin, upper

MODEL_PATH = Path(__file__).resolve().parent / "models" / "riemannian_model.joblib"
LABELS = {0: "Left Fist", 1: "Right Fist"}
CODE_TO_LABEL = {1: 0, 2: 1}
FIST_RUNS = {3, 4, 7, 8, 11, 12}


def load_model(path):
    if path.exists():
        return joblib.load(path)
    print(f"model not found at {path}, fitting one from cached epochs (this needs the dataset)", file=sys.stderr)
    from scripts.train_final_model import fit_final
    return fit_final()


def read_epochs(paths, ch_names):
    raws = []
    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=True, verbose="error")
        mne.datasets.eegbci.standardize(raw)
        raws.append(raw)
    raw = mne.concatenate_raws(raws, verbose="error")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose="error")

    if not np.isclose(raw.info["sfreq"], tfreq):
        print(f"note: resampling {raw.info['sfreq']:.0f} Hz -> {tfreq:.0f} Hz", file=sys.stderr)
        raw.resample(tfreq, verbose="error")

    missing = [c for c in ch_names if c not in raw.ch_names]
    if missing:
        raise SystemExit(f"file is missing channels the model was trained on: {missing}")
    raw.pick(ch_names)
    raw.reorder_channels(ch_names)
    raw.filter(lower, upper, fir_design="firwin", skip_by_annotation="edge", verbose="error")

    events, found = mne.events_from_annotations(raw, event_id=event_id, verbose="error")
    if len(events) == 0:
        raise SystemExit("no T1/T2 annotations found in the file(s)")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None,
                        picks="eeg", preload=True, reject_by_annotation=True, verbose="error")
    X = epochs.get_data(copy=False).astype(np.float32)
    truth = np.array([CODE_TO_LABEL[c] for c in epochs.events[:, 2]])
    onsets = epochs.events[:, 0] / raw.info["sfreq"]
    return X, truth, onsets


def warn_on_run(paths):
    for p in paths:
        m = re.search(r"R(\d{2})\.edf$", Path(p).name)
        if m and int(m.group(1)) not in FIST_RUNS:
            print(f"warning: {Path(p).name} is run {int(m.group(1))}, where T1/T2 are not left/right fist; "
                  f"labels below will not mean what the model predicts", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edf", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--no-truth", action="store_true", help="hide the T1/T2 column even if annotations exist")
    args = ap.parse_args()

    mne.set_log_level("ERROR")
    for p in args.edf:
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
    warn_on_run(args.edf)

    model = load_model(args.model)
    X, truth, onsets = read_epochs(args.edf, model.ch_names)

    scores = model.decision_function(X)
    pred = (scores > 0).astype(int)
    prob_right = 1 / (1 + np.exp(-scores))

    show_truth = not args.no_truth
    print(f"\n{len(args.edf)} file(s), {len(X)} epochs, {X.shape[1]} channels, "
          f"{(tmax - tmin):.1f}s window at {tfreq:.0f} Hz, band {model.band[0]:.0f}-{model.band[1]:.0f} Hz\n")
    head = f"{'epoch':>5}  {'onset (s)':>9}  {'p(right)':>8}  {'predicted':<10}"
    if show_truth:
        head += f"  {'annotation':<10}  {'ok'}"
    print(head)
    print("-" * len(head))
    for i, (t, p, yhat, yt) in enumerate(zip(onsets, prob_right, pred, truth)):
        line = f"{i:>5}  {t:>9.2f}  {p:>8.3f}  {LABELS[yhat]:<10}"
        if show_truth:
            line += f"  {'T1' if yt == 0 else 'T2':<10}  {'y' if yhat == yt else '.'}"
        print(line)

    print("-" * len(head))
    counts = {LABELS[k]: int((pred == k).sum()) for k in LABELS}
    print(f"predicted counts: {counts}")
    if show_truth:
        acc = (pred == truth).mean()
        print(f"accuracy vs T1/T2 annotations: {acc:.3f}  ({int((pred == truth).sum())}/{len(truth)})")


if __name__ == "__main__":
    main()

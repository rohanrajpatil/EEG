import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.results import RESULTS

STAGES = {
    1: ("PSD sensorimotor", "Stage 1  PSD (C3/Cz/C4)", "#2a78d6"),
    2: ("FBCSP 4-40Hz LOSO", "Stage 2  FBCSP (9 bands)", "#eb6834"),
    3: ("Riemannian TS 7-30Hz recentered", "Stage 3  Riemannian TS (recentered)", "#1baf7a"),
}

HEADLINE = [
    "PSD sensorimotor", "PSD all channels",
    "FBCSP 4-40Hz", "FBCSP 4-40Hz LOSO", "FBCSP 4-40Hz (9 bands)",
    "Riemannian TS 1-35Hz", "Riemannian TS 7-30Hz", "Riemannian TS 7-30Hz recentered",
    "FB-Riemannian 3-band recentered",
]


def load_all():
    frames = []
    for s in (1, 2, 3):
        p = RESULTS / f"stage{s}_metrics.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def unified_table(df):
    df = df[df.n_subjects >= 100]
    g = df.groupby(["stage", "method", "cv"], sort=False)
    t = pd.DataFrame({"test_acc": g.test_acc.mean(), "test_sd": g.test_acc.std(ddof=1),
                      "test_auc": g.test_auc.mean(), "gap": (g.train_acc.mean() - g.test_acc.mean()),
                      "folds": g.size()}).reset_index()
    return t


def pivot_by_mode(t):
    p = t.pivot_table(index=["stage", "method"], columns="cv", values="test_acc", aggfunc="first")
    cols = [c for c in ["LOSO", "GroupKFold5", "StratifiedKFold5", "within-subject StratifiedKFold5"] if c in p.columns]
    return p[cols].sort_index()


def loso_plot(path):
    d = pd.read_csv(RESULTS / "loso_distributions.csv")
    d = d.drop_duplicates(["method", "subject"], keep="last")
    series = [(d[d.method == m].test_acc.values, label, color) for m, label, color in STAGES.values()
              if (d.method == m).any()]

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax_box, ax_hist):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#b8b8b4")
        ax.spines["bottom"].set_color("#b8b8b4")
        ax.tick_params(colors="#4a4a48", labelsize=9)
        ax.grid(axis="y" if ax is ax_box else "x", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)

    labels = [lab for _, lab, _ in series]
    bp = ax_box.boxplot([v for v, _, _ in series], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#1f1f1e", linewidth=1.6),
                        whiskerprops=dict(color="#7a7a77", linewidth=1.2),
                        capprops=dict(color="#7a7a77", linewidth=1.2))
    for patch, (_, _, color) in zip(bp["boxes"], series):
        patch.set(facecolor=color, alpha=0.28, edgecolor=color, linewidth=1.4)
    rng = np.random.default_rng(0)
    for i, (v, _, color) in enumerate(series, 1):
        ax_box.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=11, color=color, alpha=0.65, linewidths=0)
        ax_box.text(i, 0.985, f"median {np.median(v):.3f}", ha="center", va="top", fontsize=8.5, color="#1f1f1e")
    ax_box.axhline(0.5, color="#7a7a77", linewidth=1, linestyle=(0, (4, 3)))
    ax_box.text(0.52, 0.505, "chance", fontsize=8, color="#4a4a48")
    ax_box.set_xticks(range(1, len(series) + 1))
    ax_box.set_xticklabels([l.split("  ")[0] for l in labels])
    ax_box.set_ylim(0.28, 1.0)
    ax_box.set_ylabel("LOSO accuracy per subject", color="#4a4a48", fontsize=9.5)
    ax_box.set_title("Per-subject spread", loc="left", fontsize=10.5, color="#1f1f1e")

    bins = np.arange(0.30, 1.001, 0.05)
    for v, lab, color in series:
        ax_hist.hist(v, bins=bins, histtype="step", linewidth=2, color=color, label=lab)
        ax_hist.hist(v, bins=bins, color=color, alpha=0.12, linewidth=0)
    ax_hist.axvline(0.5, color="#7a7a77", linewidth=1, linestyle=(0, (4, 3)))
    ax_hist.set_xlabel("LOSO accuracy per subject", color="#4a4a48", fontsize=9.5)
    ax_hist.set_ylabel("subjects", color="#4a4a48", fontsize=9.5)
    ax_hist.set_title(f"Distribution across n={len(series[0][0])} held-out subjects", loc="left",
                      fontsize=10.5, color="#1f1f1e")
    ax_hist.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor="#1f1f1e")

    fig.suptitle("Left vs right imagined fist, EEGMMIDB, leave-one-subject-out", x=0.01, ha="left",
                 fontsize=12, color="#1f1f1e", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    return [(lab, len(v), np.mean(v), np.median(v), np.std(v, ddof=1)) for v, lab, _ in series]


def main():
    df = load_all()
    t = unified_table(df)
    fmt = lambda v: f"{v:.4f}"

    print("\nunified summary, 106 subjects, all modes (mean test accuracy)\n" + "=" * 100)
    print(pivot_by_mode(t).to_string(float_format=fmt))

    print("\nheadline methods\n" + "=" * 100)
    h = t[t.method.isin(HEADLINE) & t.cv.isin(["LOSO", "GroupKFold5"])].sort_values(["cv", "stage", "test_acc"])
    print(h[["stage", "method", "cv", "test_acc", "test_sd", "test_auc", "gap", "folds"]].to_string(index=False, float_format=fmt))

    gk = t.cv == "GroupKFold5"
    sections = (
        ("negative-control channels", t[gk & t.method.str.contains("occipital|frontal|sensorimotor")]),
        ("executed -> imagined transfer", t[t.method.str.contains("exec->")]),
        ("subject-blind (leakage) splits", t[t.cv.isin(["GroupKFold5", "StratifiedKFold5"])
                                             & t.method.isin(["PSD sensorimotor", "PSD all channels", "FBCSP 4-40Hz",
                                                              "Riemannian TS 7-30Hz", "Riemannian TS 7-30Hz recentered"])]),
    )
    for title, sub in sections:
        if len(sub):
            print(f"\n{title}\n" + "=" * 100)
            print(sub[["stage", "method", "cv", "test_acc", "test_sd", "test_auc", "gap"]].to_string(index=False, float_format=fmt))

    out = RESULTS / "loso_stage_comparison.png"
    stats = loso_plot(out)
    print(f"\nLOSO per-subject summary (plotted -> {out.name})\n" + "=" * 100)
    for lab, n, mean, med, sd in stats:
        print(f"{lab:<38} n={n:>3}  mean={mean:.4f}  median={med:.4f}  sd={sd:.4f}")


if __name__ == "__main__":
    main()

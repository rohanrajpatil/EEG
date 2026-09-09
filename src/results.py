"""write results to csv files"""

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

COLUMNS = [
    "stage", "method", "cv", "n_subjects", "fold", "n_features",
    "n_test_subjects", "n_test_epochs",
    "train_acc", "test_acc", "train_auc", "test_auc"
]


def append_metrics(rows, cv, n_subjects):
    """append results to csv"""
    df_in = pd.DataFrame(rows)
    df_in["cv"] = cv
    df_in["n_subjects"] = n_subjects

    # add stage if missing
    if "stage" not in df_in.columns:
        is_psd = df_in["method"].str.contains("PSD", na=False)
        stages = np.where(is_psd, 1, 2)
        df_in["stage"] = stages

    # reorder columns
    df_out = df_in.reindex(columns=COLUMNS)

    # write to files
    RESULTS.mkdir(exist_ok=True)

    for stg, group_df in df_out.groupby("stage"):
        fname = f"stage{stg}_metrics.csv"
        fpath = RESULTS / fname
        # append or create
        if fpath.exists():
            group_df.to_csv(fpath, mode="a", header=False, index=False)
        else:
            group_df.to_csv(fpath, mode="w", header=True, index=False)

    return df_out


def summarise(df):
    """summary stats per method"""
    grouped = df.groupby(["method", "cv"], sort=False)

    summary_dict = {
        "n_subj": grouped["n_subjects"].first(),
        "n_feat": grouped["n_features"].first(),
        "folds": grouped.size(),
        "train_acc": grouped["train_acc"].mean(),
        "test_acc": grouped["test_acc"].mean(),
        "test_sd": grouped["test_acc"].std(ddof=1),
        "test_auc": grouped["test_auc"].mean(),
    }

    result_df = pd.DataFrame(summary_dict)
    result_df["gap"] = result_df["train_acc"] - result_df["test_acc"]

    return result_df.reset_index()

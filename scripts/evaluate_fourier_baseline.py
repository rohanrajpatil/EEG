import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import all_subjects, im, tfreq, load_subject_data
from src.features_psd import channels, extract_psd_features, select_channels
from src.results import append_metrics

log = logging.getLogger("baseline")


def build_model():
    """make a simple pipeline"""
    scaler = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=1000)
    pipe = Pipeline([
        ("scale", scaler),
        ("clf", clf),
    ])
    return pipe


def evaluate(X, y, groups, n_splits):
    """run cross validation"""
    cv_splitter = GroupKFold(n_splits=n_splits)
    cv_results = cross_validate(
        build_model(), X, y, groups=groups,
        cv=cv_splitter,
        scoring=["accuracy", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
    )

    fold_results = []
    for fold_idx, (_, test_indices) in enumerate(cv_splitter.split(X, y, groups)):
        fold_num = fold_idx + 1
        n_test_subj = len(np.unique(groups[test_indices]))
        n_test_epochs = len(test_indices)

        train_acc_val = cv_results["train_accuracy"][fold_idx]
        test_acc_val = cv_results["test_accuracy"][fold_idx]
        train_auc_val = cv_results["train_roc_auc"][fold_idx]
        test_auc_val = cv_results["test_roc_auc"][fold_idx]

        fold_results.append({
            "fold": fold_num,
            "n_test_subjects": n_test_subj,
            "n_test_epochs": n_test_epochs,
            "train_acc": train_acc_val,
            "test_acc": test_acc_val,
            "train_auc": train_auc_val,
            "test_auc": test_auc_val,
        })

    return pd.DataFrame(fold_results)


def report(method_name, results_df, num_features):
    """print results table"""
    print(f"\n{method_name}  ({num_features} features)")
    print("-" * 68)

    formatters_dict = {
        "train_acc": lambda v: f"{v:.4f}",
        "test_acc": lambda v: f"{v:.4f}",
        "test_auc": lambda v: f"{v:.4f}",
    }
    table_str = results_df.to_string(index=False, formatters=formatters_dict)
    print(table_str)

    print("-" * 68)
    mean_test = results_df["test_acc"].mean()
    std_test = results_df["test_acc"].std(ddof=1)
    print(f"mean test accuracy : {mean_test:.4f} +/- {std_test:.4f}")

    mean_auc = results_df["test_auc"].mean()
    print(f"mean test roc auc  : {mean_auc:.4f} +/- {results_df['test_auc'].std(ddof=1):.4f}")

    train_mean = results_df["train_acc"].mean()
    print(f"mean train accuracy: {train_mean:.4f}")


def main():
    """stage 1 baseline"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--runs", type=int, nargs="+", default=list(im))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.subjects == 0:
        subj_list = all_subjects()
    else:
        subj_list = range(1, args.subjects + 1)

    X_all_epochs, y_labels, subj_groups, channel_names = load_subject_data(subj_list, runs=args.runs)

    n_epochs = X_all_epochs.shape[0]
    n_subj = len(np.unique(subj_groups))
    n_ch = X_all_epochs.shape[1]
    n_times = X_all_epochs.shape[2]

    print(f"\nloaded {n_epochs} epochs from {n_subj} subjects | {n_ch} ch x {n_times} samples")

    n_left = int((y_labels == 0).sum())
    n_right = int((y_labels == 1).sum())
    print(f"class balance: left={n_left} right={n_right}")

    baseline_acc = max(np.bincount(y_labels)) / len(y_labels)
    print(f"majority-class baseline: {baseline_acc:.4f}")

    X_smc = extract_psd_features(
        select_channels(X_all_epochs, channel_names, channels),
        sfreq=tfreq
    )
    X_all_ch = extract_psd_features(X_all_epochs, sfreq=tfreq)

    df_smc = evaluate(X_smc, y_labels, subj_groups, args.n_splits)
    df_all = evaluate(X_all_ch, y_labels, subj_groups, args.n_splits)

    report(f"sensorimotor only {channels}", df_smc, X_smc.shape[1])
    report("all channels", df_all, X_all_ch.shape[1])

    df_smc["method"] = "PSD sensorimotor"
    df_smc["n_features"] = X_smc.shape[1]

    df_all["method"] = "PSD all channels"
    df_all["n_features"] = X_all_ch.shape[1]

    runs_str = ""
    if list(args.runs) != list(im):
        runs_str = f" runs={','.join(map(str, args.runs))}"

    all_df = pd.concat([df_smc, df_all], ignore_index=True)
    append_metrics(all_df, cv=f"GroupKFold{args.n_splits}{runs_str}",
                   n_subjects=len(np.unique(subj_groups)))


if __name__ == "__main__":
    main()

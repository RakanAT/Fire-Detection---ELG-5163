"""Leave-one-fire-out validation of the histogram classifier and the index comparison table.

Ranked metrics are used instead of accuracy: ROC-AUC, PR-AUC, lift in the top 10% and Brier score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import plots
from .bayes import fit_histogram_model, posterior_for_values


def evaluate_index(df: pd.DataFrame, index_name: str, n_bins: int = 30, alpha: float = 1.0) -> dict:
    """Score every fire with a model fitted on all the other fires, then compute metrics over all scores."""
    y_true_all = []
    y_score_all = []
    for fire_id in sorted(df["fire_id"].dropna().unique()):
        train = df[df["fire_id"] != fire_id]
        test = df[df["fire_id"] == fire_id]
        if train.empty or test.empty:
            continue
        model = fit_histogram_model(train[index_name].to_numpy(float), train["burned"].to_numpy(int), n_bins=n_bins, alpha=alpha)
        y_true_all.append(test["burned"].to_numpy(int))
        y_score_all.append(posterior_for_values(test[index_name].to_numpy(float), model))

    if not y_true_all:
        raise ValueError(f"No valid validation folds produced for index {index_name}.")
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)

    top_k = max(1, int(len(y_score) * 0.10))
    top_idx = np.argsort(y_score)[::-1][:top_k]
    baseline_rate = y_true.mean()
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    return {
        "index": index_name,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "lift_top_10": float(y_true[top_idx].mean() / baseline_rate) if baseline_rate > 0 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "calibration_probability": prob_pred,
        "calibration_observed": prob_true,
        "scores": y_score,
        "truth": y_true,
    }


def compare_indices(df: pd.DataFrame, index_names: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    results = []
    for name in index_names:
        if name not in df.columns:
            raise KeyError(f"Index column {name!r} is missing from the dataset CSV.")
        results.append(evaluate_index(df, name))
    metrics = ["roc_auc", "pr_auc", "lift_top_10", "brier_score"]
    summary = pd.DataFrame([{"index": r["index"], **{m: r[m] for m in metrics}} for r in results])
    return summary.sort_values("roc_auc", ascending=False), results


def run_comparison(dataset_csv: str, output_dir: str, index_names: list[str]) -> Path:
    """Compare the indices, print and save the table, and write all plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_csv)
    for column in ("fire_id", "burned"):
        if column not in df.columns:
            raise ValueError(f"The dataset CSV must contain a {column} column.")

    summary, results = compare_indices(df, index_names)
    summary_path = output_dir / "index_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    for name in index_names:
        plots.histogram(df, name, output_dir)
        plots.posterior(df, name, output_dir)
    plots.roc(results, output_dir)
    plots.precision_recall(results, output_dir)
    plots.calibration(results, output_dir)
    return summary_path

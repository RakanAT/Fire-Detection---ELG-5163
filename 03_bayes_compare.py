#!/usr/bin/env python3
"""Bayesian histograms for comparing vegetation-index fire predictiveness.

This script implements the traditional, non-ML workflow described in the project
brief. It fits class-conditional histograms for each vegetation index, applies a
Laplace prior and Bayes update, and evaluates the indices with leave-one-fire-out
cross-validation using ROC-AUC, PR-AUC, lift at top 10%, Brier score, and a
calibration curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vegetation indices with a Bayes-histogram model.")
    parser.add_argument("--dataset-csv", required=True, help="CSV from 02_build_dataset.py.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metric tables and plots.")
    parser.add_argument("--index-names", nargs="+", default=["ndvi", "anomaly", "dfi"], help="Vegetation indices to compare.")
    return parser.parse_args()


def build_bin_edges(values: np.ndarray, n_bins: int = 30) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite values available to build histogram bins.")
    low, high = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Non-finite values in the histogram input.")
    if low == high:
        low -= 1e-6
        high += 1e-6
    edges = np.linspace(low, high, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_histogram_model(values: np.ndarray, burned: np.ndarray, n_bins: int = 30, alpha: float = 1.0):
    values = np.asarray(values, dtype=float)
    burned = np.asarray(burned, dtype=int)
    if values.size != burned.size:
        raise ValueError("Values and burned labels must have the same length.")
    burn_vals = values[burned == 1]
    unburn_vals = values[burned == 0]
    if burn_vals.size == 0 or unburn_vals.size == 0:
        raise ValueError("Histogram model requires both burned and unburned samples.")

    edges = build_bin_edges(np.concatenate([burn_vals, unburn_vals]), n_bins=n_bins)
    burn_counts, _ = np.histogram(burn_vals, bins=edges)
    unburn_counts, _ = np.histogram(unburn_vals, bins=edges)

    burn_counts = burn_counts.astype(float) + alpha
    unburn_counts = unburn_counts.astype(float) + alpha
    prior = burn_vals.size / values.size

    burn_likelihood = burn_counts / burn_counts.sum()
    unburn_likelihood = unburn_counts / unburn_counts.sum()
    posterior_by_bin = (prior * burn_likelihood) / (prior * burn_likelihood + (1.0 - prior) * unburn_likelihood)
    return {"edges": edges, "burn_counts": burn_counts, "unburn_counts": unburn_counts, "prior": prior, "burn_likelihood": burn_likelihood, "unburn_likelihood": unburn_likelihood, "posterior_by_bin": posterior_by_bin}


def posterior_for_values(values: np.ndarray, model):
    edges = model["edges"]
    bin_idx = np.searchsorted(edges[1:-1], values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(model["posterior_by_bin"]) - 1)
    return model["posterior_by_bin"][bin_idx]


def evaluate_index(df: pd.DataFrame, index_name: str, n_bins: int = 30, alpha: float = 1.0):
    unique_fires = sorted(pd.Series(df["fire_id"].dropna().unique()).astype(int).tolist())
    y_true_all = []
    y_score_all = []

    for fire_id in unique_fires:
        train = df[df["fire_id"] != fire_id].copy()
        test = df[df["fire_id"] == fire_id].copy()
        if train.empty or test.empty:
            continue

        model = fit_histogram_model(train[index_name].to_numpy(dtype=float), train["burned"].to_numpy(dtype=int), n_bins=n_bins, alpha=alpha)
        fold_scores = posterior_for_values(test[index_name].to_numpy(dtype=float), model)
        y_true_all.append(test["burned"].to_numpy(dtype=int))
        y_score_all.append(fold_scores)

    if not y_true_all:
        raise ValueError(f"No valid validation folds produced for index {index_name}.")

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    top_k = max(1, int(len(y_score) * 0.10))
    top_idx = np.argsort(y_score)[::-1][:top_k]
    positive_rate = y_true[top_idx].mean()
    baseline_rate = y_true.mean()
    lift = positive_rate / baseline_rate if baseline_rate > 0 else float("nan")
    brier = brier_score_loss(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")

    return {
        "index": index_name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "lift_top_10": float(lift),
        "brier_score": float(brier),
        "calibration_probability": prob_pred,
        "calibration_observed": prob_true,
        "scores": y_score,
        "truth": y_true,
    }


def make_histogram_plot(df: pd.DataFrame, index_name: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in [
        ("Burned", df.loc[df["burned"] == 1, index_name].dropna()),
        ("Unburned", df.loc[df["burned"] == 0, index_name].dropna()),
    ]:
        ax.hist(values, bins=30, alpha=0.7, density=True, label=label)
    ax.set_title(f"{index_name.upper()} burned vs unburned")
    ax.set_xlabel(index_name)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"hist_{index_name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_posterior_plot(df: pd.DataFrame, index_name: str, output_dir: Path):
    model = fit_histogram_model(df[index_name].to_numpy(float), df["burned"].to_numpy(int), n_bins=30, alpha=1.0)
    # Outer edges are +/-inf, so place the two open-ended bins one bin width beyond the finite edges.
    inner = model["edges"][1:-1]
    step = inner[1] - inner[0] if len(inner) > 1 else 1.0
    centers = np.concatenate([[inner[0] - step / 2], 0.5 * (inner[1:] + inner[:-1]), [inner[-1] + step / 2]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, model["posterior_by_bin"], color="C0", linewidth=2)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_title(f"Posterior P(burned | {index_name.upper()})")
    ax.set_xlabel(index_name)
    ax.set_ylabel("Posterior probability")
    fig.tight_layout()
    path = output_dir / f"posterior_{index_name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_roc_plot(results, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        fpr, tpr, _ = roc_curve(result["truth"], result["scores"])
        ax.plot(fpr, tpr, label=f"{result['index']} (AUC={result['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "roc_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_pr_plot(results, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        precision, recall, _ = precision_recall_curve(result["truth"], result["scores"])
        ax.plot(recall, precision, label=f"{result['index']} (AP={result['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "pr_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_calibration_plot(results, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        ax.plot(result["calibration_probability"], result["calibration_observed"], marker="o", label=result["index"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "calibration.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def compare_indices(df: pd.DataFrame, index_names: list[str]):
    results = []
    for idx_name in index_names:
        if idx_name not in df.columns:
            raise KeyError(f"Index column {idx_name!r} is missing from the dataset CSV.")
        result = evaluate_index(df, idx_name)
        results.append(result)
    summary = pd.DataFrame(
        {
            "index": [r["index"] for r in results],
            "roc_auc": [r["roc_auc"] for r in results],
            "pr_auc": [r["pr_auc"] for r in results],
            "lift_top_10": [r["lift_top_10"] for r in results],
            "brier_score": [r["brier_score"] for r in results],
        }
    ).sort_values("roc_auc", ascending=False)
    return summary, results


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset_csv)
    if "fire_id" not in df.columns:
        raise ValueError("The dataset CSV must contain a fire_id column.")
    if "burned" not in df.columns:
        raise ValueError("The dataset CSV must contain the burned label column.")

    summary, results = compare_indices(df, args.index_names)
    summary_path = output_dir / "index_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    for idx_name in args.index_names:
        make_histogram_plot(df, idx_name, output_dir)
        make_posterior_plot(df, idx_name, output_dir)

    make_roc_plot(results, output_dir)
    make_pr_plot(results, output_dir)
    make_calibration_plot(results, output_dir)

    print(f"OK: {summary_path}")


if __name__ == "__main__":
    main()

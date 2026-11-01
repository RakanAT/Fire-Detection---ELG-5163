"""Figures written to the output directory.

    hist_<index>.png       burned vs unburned distribution of one index
    posterior_<index>.png  P(burned | index), i.e. the fitted histogram classifier
    roc_curve.png, pr_curve.png, calibration.png   leave-one-fire-out results for all indices
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .bayes import fit_histogram_model
from .config import INDEX_LABELS


def _label(index_name: str) -> str:
    return INDEX_LABELS.get(index_name, index_name)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def histogram(df: pd.DataFrame, index_name: str, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in [
        ("Burned", df.loc[df["burned"] == 1, index_name].dropna()),
        ("Unburned", df.loc[df["burned"] == 0, index_name].dropna()),
    ]:
        ax.hist(values, bins=30, alpha=0.7, density=True, label=label)
    ax.set_title(f"{_label(index_name)} burned vs unburned")
    ax.set_xlabel(_label(index_name))
    ax.set_ylabel("Density")
    ax.legend()
    return _save(fig, output_dir / f"hist_{index_name}.png")


def posterior(df: pd.DataFrame, index_name: str, output_dir: Path) -> Path:
    model = fit_histogram_model(df[index_name].to_numpy(float), df["burned"].to_numpy(int), n_bins=30, alpha=1.0)
    # Outer edges are +/-inf, so place the two open-ended bins one bin width beyond the finite edges.
    inner = model["edges"][1:-1]
    step = inner[1] - inner[0] if len(inner) > 1 else 1.0
    centers = np.concatenate([[inner[0] - step / 2], 0.5 * (inner[1:] + inner[:-1]), [inner[-1] + step / 2]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, model["posterior_by_bin"], color="C0", linewidth=2)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_title(f"Posterior P(burned | {_label(index_name)})")
    ax.set_xlabel(_label(index_name))
    ax.set_ylabel("Posterior probability")
    return _save(fig, output_dir / f"posterior_{index_name}.png")


def roc(results: list[dict], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        fpr, tpr, _ = roc_curve(result["truth"], result["scores"])
        ax.plot(fpr, tpr, label=f"{_label(result['index'])} (AUC={result['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    return _save(fig, output_dir / "roc_curve.png")


def precision_recall(results: list[dict], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        precision, recall, _ = precision_recall_curve(result["truth"], result["scores"])
        ax.plot(recall, precision, label=f"{_label(result['index'])} (AP={result['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend()
    return _save(fig, output_dir / "pr_curve.png")


def calibration(results: list[dict], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        ax.plot(result["calibration_probability"], result["calibration_observed"], marker="o", label=_label(result["index"]))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration curve")
    ax.legend()
    return _save(fig, output_dir / "calibration.png")

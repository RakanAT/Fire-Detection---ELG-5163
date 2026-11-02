"""Figures written to the output directory.

    hist_<index>.png        burned vs unburned distribution of one pre-fire index
    posterior_<index>.png   P(burned | index): the fitted histogram classifier, with the share of pixels per bin
    roc_curve.png           will it burn? leave-one-fire-out ROC curves of every model
    severity_roc_curve.png  how severely? the same for the more vs less severely burned half
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

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
    """P(burned | bin) on a fixed 0-1 axis, with the share of pixels in each bin as bars underneath.

    Bins holding few pixels are drawn faint: extreme probabilities there rest on little data and barely
    affect the ROC curve, which weights every pixel equally.
    """
    df = df[np.isfinite(df[index_name])]
    model = fit_histogram_model(df[index_name].to_numpy(float), df["burned"].to_numpy(int), n_bins=30, alpha=1.0)
    # Outer edges are +/-inf, so place the two open-ended bins one bin width beyond the finite edges.
    inner = model["edges"][1:-1]
    step = inner[1] - inner[0] if len(inner) > 1 else 1.0
    centers = np.concatenate([[inner[0] - step / 2], 0.5 * (inner[1:] + inner[:-1]), [inner[-1] + step / 2]])
    counts = model["burn_counts"] + model["unburn_counts"] - 2.0  # remove the Laplace pseudo-counts
    share = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.twinx()
    bars.bar(centers, 100 * share, width=step * 0.9, color="gray", alpha=0.25)
    bars.set_ylabel("Pixels in bin (%)", color="gray")
    bars.set_ylim(0, max(100 * share.max() * 3, 1))  # keep the bars in the lower third
    ax.set_zorder(bars.get_zorder() + 1)
    ax.patch.set_visible(False)

    weight = np.clip(share / share.max(), 0.08, 1.0)
    colours = np.zeros((len(centers), 4))
    colours[:, :3] = matplotlib.colors.to_rgb("C0")
    colours[:, 3] = weight
    ax.plot(centers, model["posterior_by_bin"], color="C0", linewidth=1, alpha=0.4)
    ax.scatter(centers, model["posterior_by_bin"], color=colours, s=20 + 60 * weight, zorder=3)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Posterior P(burned | {_label(index_name)}), all fires pooled\n"
                 f"faint points = bins with few pixels; grey bars = share of pixels per bin")
    ax.set_xlabel(_label(index_name))
    ax.set_ylabel("Posterior probability")
    return _save(fig, output_dir / f"posterior_{index_name}.png")


def roc(results: list[dict], output_dir: Path, filename: str = "roc_curve.png",
        title: str = "Will it burn? ROC curves (leave-one-fire-out)") -> Path:
    """Combined (naive Bayes) models are drawn thicker so they stand out from the single indices."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for result in sorted(results, key=lambda r: -r["roc_auc"]):
        fpr, tpr, _ = roc_curve(result["truth"], result["scores"])
        ax.plot(fpr, tpr, label=f"{_label(result['index'])} (AUC={result['roc_auc']:.3f})",
                linewidth=2.5 if len(result["features"]) > 1 else 1.2)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, output_dir / filename)

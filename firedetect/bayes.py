"""Bayesian histogram classifiers: one index at a time, and several combined with naive Bayes.

Fitting only counts pixels: the index range is split into bins, burned and unburned pixels are counted
per bin (plus a Laplace prior of alpha), and Bayes' rule turns the counts into P(burned | bin).

Naive Bayes combines features by assuming they are independent given the class, so their likelihoods multiply:
    P(burned | x1..xn) ∝ P(burned) * Π P(xi | burned)
It is still counting and Bayes' rule. With a single feature it gives exactly the single-index classifier.
Correlated features (e.g. NDVI and NBR) are counted as separate evidence, which makes the probabilities
overconfident; the ranking (ROC-AUC) is usually affected much less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_bin_edges(values: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Equal-width bins between the 0.5th and 99.5th percentiles; the outer bins are open-ended."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite values available to build histogram bins.")
    low, high = np.percentile(finite, [0.5, 99.5])
    if low == high:
        low -= 1e-6
        high += 1e-6
    edges = np.linspace(low, high, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def category_edges(values: np.ndarray) -> np.ndarray:
    """One bin per distinct value (for class codes such as WorldCover land cover)."""
    classes = np.unique(np.asarray(values, dtype=float)[np.isfinite(values)])
    return np.concatenate([[-np.inf], (classes[1:] + classes[:-1]) / 2, [np.inf]])


def fit_histogram_model(values: np.ndarray, burned: np.ndarray, n_bins: int = 30, alpha: float = 1.0,
                        edges: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=float)
    burned = np.asarray(burned, dtype=int)
    if values.size != burned.size:
        raise ValueError("Values and burned labels must have the same length.")
    burn_vals = values[burned == 1]
    unburn_vals = values[burned == 0]
    if burn_vals.size == 0 or unburn_vals.size == 0:
        raise ValueError("Histogram model requires both burned and unburned samples.")

    edges = build_bin_edges(values, n_bins=n_bins) if edges is None else edges
    burn_counts = np.histogram(burn_vals, bins=edges)[0] + alpha
    unburn_counts = np.histogram(unburn_vals, bins=edges)[0] + alpha
    prior = burn_vals.size / values.size

    burn_likelihood = burn_counts / burn_counts.sum()  # P(bin | burned)
    unburn_likelihood = unburn_counts / unburn_counts.sum()  # P(bin | unburned)
    posterior_by_bin = (prior * burn_likelihood) / (prior * burn_likelihood + (1.0 - prior) * unburn_likelihood)
    return {
        "edges": edges,
        "burn_counts": burn_counts,
        "unburn_counts": unburn_counts,
        "prior": prior,
        "burn_likelihood": burn_likelihood,
        "unburn_likelihood": unburn_likelihood,
        "posterior_by_bin": posterior_by_bin,
    }


def _bin_index(values: np.ndarray, model: dict) -> np.ndarray:
    bin_idx = np.searchsorted(model["edges"][1:-1], values, side="right")
    return np.clip(bin_idx, 0, len(model["posterior_by_bin"]) - 1)


def fit_naive_bayes(frame: pd.DataFrame, features: list[str], labels: np.ndarray, n_bins: int = 30, alpha: float = 1.0,
                    categorical: list[str] = ()) -> dict:
    """One histogram model per feature, fitted on the same pixels, plus the overall prior."""
    labels = np.asarray(labels, dtype=int)
    models = {}
    for name in features:
        values = frame[name].to_numpy(float)
        edges = category_edges(values) if name in categorical else None
        models[name] = fit_histogram_model(values, labels, n_bins, alpha, edges)
    return {"prior": float(labels.mean()), "features": models}


def naive_bayes_posterior(frame: pd.DataFrame, model: dict) -> np.ndarray:
    """P(burned | all features): prior odds times the product of each feature's likelihood ratio."""
    prior = model["prior"]
    log_odds = np.full(len(frame), np.log(prior / (1.0 - prior)))
    for name, feature_model in model["features"].items():
        idx = _bin_index(frame[name].to_numpy(float), feature_model)
        log_odds += np.log(feature_model["burn_likelihood"][idx]) - np.log(feature_model["unburn_likelihood"][idx])
    return 1.0 / (1.0 + np.exp(-log_odds))

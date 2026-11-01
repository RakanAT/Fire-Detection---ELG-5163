"""Bayesian histogram classifier for a single index.

Fitting only counts pixels: the index range is split into bins, burned and unburned pixels are counted
per bin (plus a Laplace prior of alpha), and Bayes' rule turns the counts into P(burned | bin).
"""

from __future__ import annotations

import numpy as np


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


def fit_histogram_model(values: np.ndarray, burned: np.ndarray, n_bins: int = 30, alpha: float = 1.0) -> dict:
    values = np.asarray(values, dtype=float)
    burned = np.asarray(burned, dtype=int)
    if values.size != burned.size:
        raise ValueError("Values and burned labels must have the same length.")
    burn_vals = values[burned == 1]
    unburn_vals = values[burned == 0]
    if burn_vals.size == 0 or unburn_vals.size == 0:
        raise ValueError("Histogram model requires both burned and unburned samples.")

    edges = build_bin_edges(values, n_bins=n_bins)
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


def posterior_for_values(values: np.ndarray, model: dict) -> np.ndarray:
    """Look up P(burned | bin) for each value."""
    bin_idx = np.searchsorted(model["edges"][1:-1], values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(model["posterior_by_bin"]) - 1)
    return model["posterior_by_bin"][bin_idx]

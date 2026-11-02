"""Leave-one-fire-out validation of the histogram classifiers and the result tables.

Each table answers one finding (see README "Findings"):
    index_comparison.csv      will an area burn? pre-fire indices, their combination, the landscape, the dNDVI control
    near_vs_far.csv           the same against unburned vegetation next to the fire and 1-3 km away
    landscape_comparison.csv  each landscape factor on its own, and which way it points
    severity_comparison.csv   do pre-fire indices predict how severely an area burns?
    per_fire.csv              data checks per fire (already-burned "pre-fire" scene, weak post-fire scar)
Ranked metrics are used instead of accuracy: ROC-AUC and lift in the top 10%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import plots
from .bayes import build_bin_edges, category_edges
from .config import (CATEGORICAL_FEATURES, COMBINED_MODELS, CONTROL_INDEX, LANDSCAPE_FACTORS, PRE_FIRE_INDICES,
                     RBR_OFFSET, WEAK_SCAR_AUC)


def _leave_one_fire_out(df: pd.DataFrame, features: list[str], label: str, n_bins: int, alpha: float):
    """Scores of every row from the naive Bayes model fitted on all *other* fires, without refitting per fire.

    Each fire's class counts per bin are computed once; the model for a held-out fire uses the total counts minus
    that fire's counts, which is exactly what fitting on the other fires would count. Bin edges are set once from
    all rows' values (they use no labels). Returns (scores, usable rows).
    """
    fire_idx = np.unique(df["fire_id"].to_numpy(), return_inverse=True)[1]
    n_fires = fire_idx.max() + 1
    y = df[label].to_numpy(int)
    class_counts = np.bincount(fire_idx * 2 + y, minlength=n_fires * 2).reshape(n_fires, 2)
    train_classes = class_counts.sum(axis=0) - class_counts
    usable_fold = (train_classes > 0).all(axis=1)
    prior = np.clip(train_classes[:, 1] / np.maximum(train_classes.sum(axis=1), 1), 1e-9, 1 - 1e-9)
    log_odds = np.log(prior / (1 - prior))[fire_idx]

    for feature in features:
        values = df[feature].to_numpy(float)
        edges = category_edges(values) if feature in CATEGORICAL_FEATURES else build_bin_edges(values, n_bins)
        n_edges = len(edges) - 1
        bins = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, n_edges - 1)
        counts = np.bincount((fire_idx * n_edges + bins) * 2 + y, minlength=n_fires * n_edges * 2).reshape(n_fires, n_edges, 2)
        train = counts.sum(axis=0) - counts + alpha
        likelihood = train / train.sum(axis=1, keepdims=True)  # P(bin | class), per held-out fire
        log_odds += (np.log(likelihood[..., 1]) - np.log(likelihood[..., 0]))[fire_idx, bins]
    return 1.0 / (1.0 + np.exp(-log_odds)), usable_fold[fire_idx]


def evaluate_model(df: pd.DataFrame, name: str, features: list[str], label: str = "burned", n_bins: int = 30,
                   alpha: float = 1.0) -> dict:
    """Score every fire with a model fitted on all the other fires, then compute metrics over all scores.

    One feature is the single-index histogram classifier; several are combined with naive Bayes. Rows where any
    feature is missing (e.g. dNDVI under post-fire cloud) are left out for this model only.
    """
    df = df[np.isfinite(df[features]).all(axis=1)]
    if df.empty:
        raise ValueError(f"No rows with all features for {name}.")
    scores, usable = _leave_one_fire_out(df, features, label, n_bins, alpha)
    if not usable.any():
        raise ValueError(f"No valid validation folds produced for {name}.")
    y_true = df[label].to_numpy(int)[usable]
    y_score = scores[usable]
    top = np.argsort(y_score)[::-1][:max(1, len(y_score) // 10)]
    fire_column = "fire_label" if "fire_label" in df else "fire_id"
    return {
        "index": name,
        "features": features,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "lift_top_10": float(y_true[top].mean() / y_true.mean()),
        "scores": y_score,
        "truth": y_true,
        "fires": df[fire_column].astype(str).to_numpy()[usable],
    }


def compare_models(df: pd.DataFrame, models: dict[str, list[str]], label: str = "burned") -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate each named model (name -> features) and return a summary table sorted by ROC-AUC."""
    results = [evaluate_model(df, name, features, label) for name, features in models.items()]
    summary = pd.DataFrame([{"index": r["index"], "roc_auc": round(r["roc_auc"], 3), "lift_top_10": round(r["lift_top_10"], 2),
                             "pixels": len(r["truth"])} for r in results])
    return summary.sort_values("roc_auc", ascending=False), results


def compare_indices(df: pd.DataFrame, index_names: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Each index on its own."""
    return compare_models(df, {name: [name] for name in index_names})


def available_models(df: pd.DataFrame, index_names: list[str]) -> dict[str, list[str]]:
    """Single indices with data, plus the naive Bayes combinations whose features all have data."""
    has_data = {c for c in df.columns if df[c].notna().any()}
    models = {name: [name] for name in index_names if name in has_data}
    models.update({name: features for name, features in COMBINED_MODELS.items() if set(features) <= has_data})
    return models


def severity_table(df: pd.DataFrame, models: dict[str, list[str]]) -> tuple[pd.DataFrame, list[dict], float]:
    """Within burned pixels: do pre-fire indices predict the more severely burned half (RBR above its median)?

    Returns the summary, the results and the RBR median used as the split.
    """
    burned = df[(df["burned"] == 1) & np.isfinite(df["dnbr"])].copy()
    burned["rbr"] = burned["dnbr"] / (burned["nbr"] + RBR_OFFSET)
    split = float(burned["rbr"].median())
    burned["high_severity"] = (burned["rbr"] > split).astype(int)
    summary, results = compare_models(burned, models, label="high_severity")
    return summary, results, split


def landscape_table(df: pd.DataFrame) -> pd.DataFrame:
    """Each landscape factor on its own: ROC-AUC, and its median for burned and unburned pixels (the direction)."""
    factors = [name for name in LANDSCAPE_FACTORS if df[name].notna().any()]
    summary, _ = compare_indices(df, factors)
    medians = df.groupby("burned")[factors].median().T.round(1)
    medians.columns = ["median_unburned", "median_burned"]
    return summary.merge(medians, left_on="index", right_index=True)


def near_far_table(df: pd.DataFrame, models: dict[str, list[str]]) -> pd.DataFrame:
    """ROC-AUC of burned vs unburned 0-250 m away (near) and 1-3 km away (far), on the same 20 m context scenes."""
    columns = {}
    for zone in ("near", "far"):
        subset = df[df["zone"].isin(["burned", zone])].reset_index(drop=True)
        columns[f"roc_auc_burned_vs_{zone}"] = compare_models(subset, models)[0].set_index("index")["roc_auc"]
    return pd.DataFrame(columns).reindex(list(models)).reset_index()


def per_fire_table(results: list[dict], summary_csv: Path) -> pd.DataFrame:
    """ROC-AUC of each held-out fire's scores, per model, joined with the build stage's per-fire summary.

    weak_post_fire_scar marks fires whose own dNDVI separates burned from unburned poorly: a doubtful perimeter
    (or a post-fire scene that missed the scar).
    """
    rows = {}
    for result in results:
        frame = pd.DataFrame({"fire_label": result["fires"], "truth": result["truth"], "score": result["scores"]})
        for fire, group in frame.groupby("fire_label"):
            auc = roc_auc_score(group["truth"], group["score"]) if group["truth"].nunique() == 2 else float("nan")
            rows.setdefault(fire, {"fire_label": fire})[f"roc_auc_{result['index']}"] = round(auc, 3)
    table = pd.DataFrame(list(rows.values()))
    if f"roc_auc_{CONTROL_INDEX}" in table:
        table["weak_post_fire_scar"] = table[f"roc_auc_{CONTROL_INDEX}"] < WEAK_SCAR_AUC
    if Path(summary_csv).exists():
        table = pd.read_csv(summary_csv, dtype={"fire_label": str}).merge(table, on="fire_label", how="right")
    flags = [c for c in ("suspect_pre_burned", "weak_post_fire_scar") if c in table]
    return table.sort_values([*flags, "fire_label"], ascending=[False] * len(flags) + [True])


def _write(table: pd.DataFrame, path: Path, title: str) -> None:
    table.to_csv(path, index=False)
    print(f"\n{title} -> {path.name}")
    print(table.to_string(index=False))


def run_comparison(dataset_csv: str, context_csv: str, output_dir: str, index_names: list[str], fire_summary_csv: str) -> Path:
    """Compare the indices, print and save all tables, and write all plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_csv, dtype={"fire_label": str})
    index_names = [name for name in index_names if name in df and df[name].notna().any()]

    # 1 and 2: will an area burn? Single indices, their combination, the landscape, and the dNDVI control.
    models = available_models(df, index_names)
    summary, results = compare_models(df, models)
    summary_path = output_dir / "index_comparison.csv"
    _write(summary, summary_path, "Will it burn? Burned vs 0-250 m ring, leave-one-fire-out")
    plots.roc(results, output_dir)
    per_fire_table(results, fire_summary_csv).to_csv(output_dir / "per_fire.csv", index=False)

    prefire_models = {name: features for name, features in models.items() if CONTROL_INDEX not in features}
    context = pd.read_csv(context_csv, dtype={"fire_label": str})
    _write(near_far_table(context, {name: f for name, f in available_models(context, PRE_FIRE_INDICES).items() if name in prefire_models}),
           output_dir / "near_vs_far.csv", "Will it burn? Burned vs unburned 0-250 m (near) and 1-3 km (far) away, 20 m scenes")
    _write(landscape_table(df), output_dir / "landscape_comparison.csv", "Landscape factors on their own")

    # 3: how severely will it burn?
    severity_summary, severity_results, split = severity_table(df, prefire_models)
    _write(severity_summary, output_dir / "severity_comparison.csv",
           f"How severely? More vs less severely burned half (RBR above / below {split:.3f}), burned pixels, leave-one-fire-out")
    plots.roc(severity_results, output_dir, "severity_roc_curve.png", "ROC curves: predicting the more severely burned half")

    # Why the indices fail: burned vs unburned distribution and the fitted classifier, per pre-fire index.
    for old_plot in [*output_dir.glob("hist_*.png"), *output_dir.glob("posterior_*.png")]:
        old_plot.unlink()
    for name in index_names:
        if name in PRE_FIRE_INDICES:
            plots.histogram(df, name, output_dir)
            plots.posterior(df, name, output_dir)
    return summary_path

"""Per-fire maps of what each index flagged as likely to burn, against what actually burned.

Each fire is scored with the histogram classifier fitted on all *other* fires (the same leave-one-fire-out
models behind the ROC curves), so the maps show genuine predictions. A pixel is flagged when
P(burned | index value) is above the training share of burned pixels (~0.5). Only analysed pixels are coloured:
the eroded perimeter (burned) and the unburned ring, after cloud, land-cover and baseline masking. The index
panels are drawn over shaded relief (from the DEM) so ridges and valleys around the perimeter are visible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

from . import terrain
from .bayes import fit_naive_bayes, naive_bayes_posterior
from .config import BANDS, CATEGORICAL_FEATURES, INDEX_LABELS, PRE_FIRE_INDICES, SceneSet
from .dataset import Scene, load_perimeters, load_scene

LOGGER = logging.getLogger(__name__)

FLAG_RULE = "P(burned) above the training share of burned pixels"

# Outcome codes per analysed pixel and their colours.
OUTCOMES = [
    ("Flagged, burned (hit)", "#d62728"),
    ("Flagged, did not burn (false alarm)", "#ff9f1c"),
    ("Not flagged, burned (missed)", "#1f4e9c"),
    ("Not flagged, did not burn", "#d9d9d9"),
]


def _false_colour(bands: np.ndarray) -> np.ndarray:
    """SWIR2 / NIR / red composite: healthy vegetation green, bare soil pink, burn scars dark red."""
    rgb = np.dstack([bands[BANDS.index(b)] for b in ("B12", "B08", "B04")])
    has_data = bands[BANDS.index("dataMask")] > 0
    scale = np.percentile(rgb[has_data], 98, axis=0) if has_data.any() else np.ones(3)
    return np.clip(rgb / np.maximum(scale, 1), 0, 1)


def _crop(mask: np.ndarray, pad: int = 10) -> tuple[slice, slice]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return slice(None), slice(None)
    return (slice(max(rows.min() - pad, 0), rows.max() + pad + 1), slice(max(cols.min() - pad, 0), cols.max() + pad + 1))


def fire_map(scene: Scene, dataset: pd.DataFrame, index_names: list[str], ring_m: float, out_path: Path) -> Path:
    ring = scene.zone_mask(0, ring_m)
    window = _crop(scene.fire_mask | ring)
    images = [("Pre-fire scene", scene.bands)] + ([("Post-fire scene (5-45 days after)", scene.postfire_bands)]
                                                  if scene.postfire_bands is not None else [])
    if scene.elevation is not None:
        background, background_note = terrain.hillshade(scene.elevation, scene.pixel_size_m), "over shaded relief"
    else:
        background, background_note = _false_colour(scene.bands).mean(axis=2), "over the pre-fire image"
    train = dataset[dataset["fire_label"] != scene.fire_key]
    panels = [(name, [name]) for name in index_names]
    if all(name in index_names for name in PRE_FIRE_INDICES):
        panels.append(("combined", PRE_FIRE_INDICES))

    n_panels = len(images) + len(panels)
    n_cols = min(n_panels, 6)
    n_rows = -(-n_panels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 6.0 * n_rows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    for ax, (title, bands) in zip(axes, images):
        ax.imshow(_false_colour(bands)[window])
        ax.contour(scene.fire_mask[window], levels=[0.5], colors="cyan", linewidths=1)
        ax.set_title(f"{title}\nfalse colour (SWIR2, NIR, red)\ncyan = EFFIS perimeter", fontsize=9)

    for ax, (name, features) in zip(axes[len(images):], panels):
        layers = {feature: getattr(scene, feature) for feature in features}
        analysed = scene.burned_mask | ring
        for layer in layers.values():
            analysed &= np.isfinite(layer)
        truth = scene.burned_mask[analysed]
        train_rows = train[np.isfinite(train[features]).all(axis=1)]
        model = fit_naive_bayes(train_rows, features, train_rows["burned"].to_numpy(int), categorical=CATEGORICAL_FEATURES)
        posterior = naive_bayes_posterior(pd.DataFrame({f: layer[analysed] for f, layer in layers.items()}), model)
        flagged = posterior > model["prior"]

        outcome = np.full(scene.fire_mask.shape, np.nan)
        outcome[analysed] = np.select([flagged & truth, flagged & ~truth, ~flagged & truth], [0, 1, 2], default=3)
        ax.imshow(background[window], cmap="gray", vmin=0, vmax=1)
        ax.imshow(outcome[window], cmap=ListedColormap([c for _, c in OUTCOMES]), vmin=-0.5, vmax=3.5,
                  interpolation="nearest", alpha=0.85)
        ax.contour(scene.fire_mask[window], levels=[0.5], colors="black", linewidths=0.8)

        hit_rate = 100 * flagged[truth].mean() if truth.any() else float("nan")
        false_alarm = 100 * flagged[~truth].mean() if (~truth).any() else float("nan")
        auc = roc_auc_score(truth, posterior) if 0 < truth.sum() < truth.size else float("nan")
        ax.set_title(f"{INDEX_LABELS.get(name, name)}\nflagged {100 * flagged.mean():.0f}% of pixels, ROC-AUC {auc:.2f}\n"
                     f"burned pixels flagged {hit_rate:.0f}%, unburned flagged {false_alarm:.0f}%", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=label) for label, c in OUTCOMES], loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Fire {scene.fire_key} ({scene.path.stem.split('__')[1].replace('_', ' to ')} pre-fire window). "
                 f"Flagged = {FLAG_RULE}, model fitted on all other fires; outcomes drawn {background_note}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), h_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def write_fire_maps(dataset_csv: str, fires_file: str, scene_set: SceneSet, output_dir: str, index_names: list[str],
                    ring_m: float, fire_ids: list[str] | None = None, erode_m: float = 20.0) -> Path:
    """One map per fire in the dataset (or only fire_ids), written to output_dir/fire_maps/<fire id>.png."""
    dataset = pd.read_csv(dataset_csv, dtype={"fire_label": str})
    index_names = [name for name in index_names if name in dataset and dataset[name].notna().any()]
    perimeters = load_perimeters(fires_file)
    images = dataset.drop_duplicates("fire_label").set_index("fire_label")["source_image"]
    if fire_ids:
        images = images[images.index.isin([str(f) for f in fire_ids])]
    map_dir = Path(output_dir) / "fire_maps"
    if not fire_ids:
        for old_map in map_dir.glob("*.png"):
            old_map.unlink()

    for fire_key, image_name in images.items():
        tif_path = scene_set.prefire_dir / image_name
        scene = load_scene(tif_path, perimeters, scene_set, erode_m) if tif_path.exists() else None
        if scene is None:
            LOGGER.warning("Could not load the scene of fire %s; no map.", fire_key)
            continue
        LOGGER.info("Map written: %s", fire_map(scene, dataset, index_names, ring_m, map_dir / f"{fire_key}.png"))
    return map_dir

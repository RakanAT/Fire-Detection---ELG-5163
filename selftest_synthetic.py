import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    module_path = Path(__file__).with_name("03_bayes_compare.py")
    spec = importlib.util.spec_from_file_location("bayes_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def generate_synthetic_data():
    rng = np.random.default_rng(42)
    rows = []
    for fire_id in range(1, 7):
        for burned in [0, 1]:
            n = 400
            if burned:
                dfi = rng.normal(0.80, 0.08, size=n)
                ndvi = rng.normal(0.50, 0.12, size=n)
                anomaly = rng.normal(0.15, 0.08, size=n)
            else:
                dfi = rng.normal(0.25, 0.10, size=n)
                ndvi = rng.normal(0.48, 0.12, size=n)
                anomaly = rng.normal(-0.02, 0.10, size=n)
            for i in range(n):
                rows.append({
                    "fire_id": fire_id,
                    "burned": burned,
                    "ndvi": float(ndvi[i]),
                    "anomaly": float(anomaly[i]),
                    "dfi": float(dfi[i]),
                })
    return pd.DataFrame(rows)


def run():
    module = load_module()
    df = generate_synthetic_data()
    summary, results = module.compare_indices(df, ["ndvi", "anomaly", "dfi"])
    ranking = summary["index"].tolist()
    dfi_rank = ranking.index("dfi")
    ndvi_rank = ranking.index("ndvi")
    if dfi_rank >= ndvi_rank:
        raise AssertionError(f"DFI was not ranked ahead of NDVI. Ranking: {ranking}; metrics: {summary.to_dict(orient='records')}")
    if summary.loc[summary["index"] == "dfi", "roc_auc"].iloc[0] <= summary.loc[summary["index"] == "ndvi", "roc_auc"].iloc[0]:
        raise AssertionError(f"DFI ROC-AUC should exceed NDVI on the synthetic dataset. {summary.to_dict(orient='records')}")
    print(summary.to_string(index=False))
    print("PASS: DFI outperforms NDVI on synthetic burned-area signal.")


if __name__ == "__main__":
    run()

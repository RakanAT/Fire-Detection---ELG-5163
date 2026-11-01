"""Synthetic check that the comparison ranks an informative index (DFI) above a weak one (NDVI).

Run with:  python main.py --stages selftest   or   python -m tests.selftest_synthetic
"""

import numpy as np
import pandas as pd

from firedetect.evaluate import compare_indices


def generate_synthetic_data():
    rng = np.random.default_rng(42)
    rows = []
    for fire_id in range(1, 7):
        for burned in [0, 1]:
            n = 400
            if burned:
                dfi = rng.normal(0.80, 0.08, size=n)
                ndvi = rng.normal(0.50, 0.12, size=n)
                temporal_ndvi = rng.normal(0.15, 0.08, size=n)
            else:
                dfi = rng.normal(0.25, 0.10, size=n)
                ndvi = rng.normal(0.48, 0.12, size=n)
                temporal_ndvi = rng.normal(-0.02, 0.10, size=n)
            for i in range(n):
                rows.append({
                    "fire_id": fire_id,
                    "burned": burned,
                    "ndvi": float(ndvi[i]),
                    "temporal_ndvi": float(temporal_ndvi[i]),
                    "dfi": float(dfi[i]),
                })
    return pd.DataFrame(rows)


def run():
    df = generate_synthetic_data()
    summary, _ = compare_indices(df, ["ndvi", "temporal_ndvi", "dfi"])
    ranking = summary["index"].tolist()
    if ranking.index("dfi") >= ranking.index("ndvi"):
        raise AssertionError(f"DFI was not ranked ahead of NDVI. Ranking: {ranking}; metrics: {summary.to_dict(orient='records')}")
    if summary.loc[summary["index"] == "dfi", "roc_auc"].iloc[0] <= summary.loc[summary["index"] == "ndvi", "roc_auc"].iloc[0]:
        raise AssertionError(f"DFI ROC-AUC should exceed NDVI on the synthetic dataset. {summary.to_dict(orient='records')}")
    print(summary.to_string(index=False))
    print("PASS: DFI outperforms NDVI on synthetic burned-area signal.")


if __name__ == "__main__":
    run()

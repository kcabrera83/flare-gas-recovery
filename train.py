import json
import os
import pickle

import numpy as np

from flare_gas_recovery.data_generator import generate_synthetic_data
from flare_gas_recovery.utils.preprocessor import (
    FEATURE_COLS,
    EMISSION_TARGET,
    RECOVERY_TARGET,
    SAVINGS_TARGET,
    preprocess,
)
from flare_gas_recovery.models.recovery_optimizer import RecoveryOptimizer
from flare_gas_recovery.models.emission_predictor import EmissionPredictor

OUTPUT_DIR = os.path.join("outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  FLARE GAS RECOVERY - MODEL TRAINING (Dask + Polars)")
    print("=" * 60)

    import polars as pl
    print("\n[1/5] Generating synthetic dataset (5000 samples)...")
    df_pd = generate_synthetic_data(n_samples=5000)
    df = pl.from_pandas(df_pd)
    csv_path = os.path.join(OUTPUT_DIR, "dataset.csv")
    df_pd.to_csv(csv_path, index=False)
    print(f"  Dataset saved to {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    print("\n  Polars summary statistics:")
    print(f"  {df.describe()}")

    print("\n[2/5] Preprocessing for recovery_rate_mcf (Dask + Polars)...")
    data_r = preprocess(df_pd, RECOVERY_TARGET)
    print(f"  Train: {data_r['X_train'].shape}, Test: {data_r['X_test'].shape}")

    print("[3/5] Training Dask-ML LinearRegression (recovery_rate_mcf)...")
    optimizer = RecoveryOptimizer(target_name=RECOVERY_TARGET)
    optimizer.fit(data_r["X_train"], data_r["y_train"])
    metrics_r = optimizer.evaluate(data_r["X_test"], data_r["y_test"])
    importances_r = optimizer.feature_importances(FEATURE_COLS)
    print(f"  Metrics: {json.dumps(metrics_r, indent=4)}")
    print(f"  Feature importances: {json.dumps(importances_r, indent=4)}")

    print("\n[3b/5] Training Dask-ML LinearRegression (economic_savings_usd)...")
    data_s = preprocess(df_pd, SAVINGS_TARGET)
    savings_optimizer = RecoveryOptimizer(target_name=SAVINGS_TARGET)
    savings_optimizer.fit(data_s["X_train"], data_s["y_train"])
    metrics_s = savings_optimizer.evaluate(data_s["X_test"], data_s["y_test"])
    print(f"  Metrics: {json.dumps(metrics_s, indent=4)}")

    print("\n[4/5] Preprocessing for co2_emissions_tons...")
    data_e = preprocess(df_pd, EMISSION_TARGET)
    print(f"  Train: {data_e['X_train'].shape}, Test: {data_e['X_test'].shape}")

    print("  Training Dask-ML LinearRegression (co2_emissions_tons)...")
    emitter = EmissionPredictor()
    emitter.fit(data_e["X_train"], data_e["y_train"])
    metrics_e = emitter.evaluate(data_e["X_test"], data_e["y_test"])
    importances_e = emitter.feature_importances(FEATURE_COLS)
    print(f"  Metrics: {json.dumps(metrics_e, indent=4)}")
    print(f"  Feature importances: {json.dumps(importances_e, indent=4)}")

    print("\n[5/5] Saving models...")
    paths = {
        "recovery_optimizer": os.path.join(OUTPUT_DIR, "recovery_optimizer.pkl"),
        "savings_optimizer": os.path.join(OUTPUT_DIR, "savings_optimizer.pkl"),
        "emission_predictor": os.path.join(OUTPUT_DIR, "emission_predictor.pkl"),
    }

    with open(paths["recovery_optimizer"], "wb") as f:
        pickle.dump(optimizer, f)
    with open(paths["savings_optimizer"], "wb") as f:
        pickle.dump(savings_optimizer, f)
    with open(paths["emission_predictor"], "wb") as f:
        pickle.dump(emitter, f)

    summary = {
        "recovery_rate_mcf": metrics_r,
        "economic_savings_usd": metrics_s,
        "co2_emissions_tons": metrics_e,
        "feature_importances_recovery": importances_r,
        "feature_importances_emission": importances_e,
        "feature_columns": FEATURE_COLS,
        "n_samples": len(df_pd),
        "framework": "dask/polars",
    }
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

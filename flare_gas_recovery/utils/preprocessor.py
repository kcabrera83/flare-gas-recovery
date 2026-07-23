import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "flare_gas_rate_mcf",
    "methane_pct",
    "ethane_pct",
    "propane_pct",
    "butane_pct",
    "ambient_temp_c",
    "wind_speed_ms",
    "steam_injection_rate",
    "flare_efficiency_pct",
]

RECOVERY_TARGET = "recovery_rate_mcf"
EMISSION_TARGET = "co2_emissions_tons"
SAVINGS_TARGET = "economic_savings_usd"


def preprocess(df, target_col, test_size=0.2, random_state=42):
    """Split features and target into train/test sets.

    Scaling is handled internally by Dask-ML models.
    """
    X = df[FEATURE_COLS].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_COLS,
    }

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    """Scale features and split into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str
        Column name for the regression target.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    X = df[FEATURE_COLS].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
    }

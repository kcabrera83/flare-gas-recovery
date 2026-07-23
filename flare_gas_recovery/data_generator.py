import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=5000, random_state=42):
    """Generate synthetic flare gas recovery dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with flare gas recovery features and targets.
    """
    rng = np.random.RandomState(random_state)

    flare_gas_rate_mcf = rng.uniform(100, 5000, n_samples)
    methane_pct = rng.uniform(40, 90, n_samples)
    ethane_pct = rng.uniform(5, 25, n_samples)
    propane_pct = rng.uniform(2, 15, n_samples)
    butane_pct = rng.uniform(1, 8, n_samples)

    ambient_temp_c = rng.uniform(-10, 45, n_samples)
    wind_speed_ms = rng.uniform(0, 25, n_samples)
    steam_injection_rate = rng.uniform(0.1, 2.0, n_samples)

    flare_efficiency_pct = (
        60
        + 0.002 * flare_gas_rate_mcf
        + 0.05 * methane_pct
        + 0.3 * steam_injection_rate
        - 0.1 * wind_speed_ms
        + 0.02 * ambient_temp_c
        + rng.normal(0, 3, n_samples)
    )
    flare_efficiency_pct = np.clip(flare_efficiency_pct, 30, 99)

    recovery_rate_mcf = (
        flare_gas_rate_mcf
        * (flare_efficiency_pct / 100)
        * (0.3 + 0.004 * methane_pct + 0.01 * steam_injection_rate)
        + rng.normal(0, 20, n_samples)
    )
    recovery_rate_mcf = np.clip(recovery_rate_mcf, 0, flare_gas_rate_mcf * 0.95)

    gas_value_per_mcf = (
        0.03 * methane_pct
        + 0.05 * ethane_pct
        + 0.07 * propane_pct
        + 0.10 * butane_pct
    )
    economic_savings_usd = (
        recovery_rate_mcf * gas_value_per_mcf * 365
        + 0.5 * recovery_rate_mcf * steam_injection_rate * 100
        + rng.normal(0, 500, n_samples)
    )
    economic_savings_usd = np.clip(economic_savings_usd, 0, None)

    co2_emissions_tons = (
        flare_gas_rate_mcf * 0.001 * (1 - flare_efficiency_pct / 100) * 365
        + 0.05 * wind_speed_ms
        - 0.3 * steam_injection_rate
        + 0.02 * ambient_temp_c
        + rng.normal(0, 5, n_samples)
    )
    co2_emissions_tons = np.clip(co2_emissions_tons, 0, None)

    df = pd.DataFrame(
        {
            "flare_gas_rate_mcf": flare_gas_rate_mcf,
            "methane_pct": methane_pct,
            "ethane_pct": ethane_pct,
            "propane_pct": propane_pct,
            "butane_pct": butane_pct,
            "ambient_temp_c": ambient_temp_c,
            "wind_speed_ms": wind_speed_ms,
            "steam_injection_rate": steam_injection_rate,
            "flare_efficiency_pct": flare_efficiency_pct,
            "recovery_rate_mcf": recovery_rate_mcf,
            "co2_emissions_tons": co2_emissions_tons,
            "economic_savings_usd": economic_savings_usd,
        }
    )

    return df

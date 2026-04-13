import numpy as np
import pandas as pd

def generate_energy_data(start_date, end_date, freq="h", seed=42):

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    df = pd.DataFrame({"timestamp": timestamps})

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Temperature
    annual = 2 * np.pi * df["day_of_year"] / 365
    daily = 2 * np.pi * df["hour"] / 24

    df["temperature_c"] = (
        24
        + 8 * np.sin(annual)
        + 3 * np.sin(daily)
        + rng.normal(0, 1.5, len(df))
    )

    # Occupancy
    business = ((df["hour"] >= 8) & (df["hour"] <= 18) & (df["day_of_week"] < 5)).astype(int)
    df["occupancy_index"] = 0.2 + 0.7 * business + 0.1 * rng.random(len(df))

    # Production
    df["production_index"] = 60 + 25 * business + rng.normal(0, 3, len(df))

    # Energy
    df["energy_kwh"] = (
        200
        + df["temperature_c"] * 2
        + df["occupancy_index"] * 50
        + df["production_index"] * 2
        + rng.normal(0, 10, len(df))
    )

    return df
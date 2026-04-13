import numpy as np

def create_features(df):

    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Cyclical features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Lag features
    df["lag_1"] = df["energy_kwh"].shift(1)
    df["lag_24"] = df["energy_kwh"].shift(24)

    # Rolling
    df["rolling_mean_24"] = df["energy_kwh"].shift(1).rolling(24).mean()

    df = df.dropna()

    return df
import pandas as pd

def forecast(model, df, feature_cols, steps=24):

    future_preds = []
    df = df.copy()

    for _ in range(steps):

        last_row = df.iloc[-1:].copy()

        pred = model.predict(last_row[feature_cols])[0]
        future_preds.append(pred)

        new_row = last_row.copy()
        new_row["energy_kwh"] = pred
        new_row["timestamp"] = new_row["timestamp"] + pd.Timedelta(hours=1)

        df = pd.concat([df, new_row], ignore_index=True)

    return future_preds
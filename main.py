import pandas as pd

from src.config import *
from src.data_generation import generate_energy_data
from src.features import create_features
from src.model import train_models, select_best, save_model
from src.visualize import plot_results

def main():

    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    print("Generating data...")
    df = generate_energy_data(START_DATE, END_DATE, FREQ)
    df.to_csv(RAW_DATA_FILE, index=False)

    print("Feature engineering...")
    df = create_features(df)

    feature_cols = [col for col in df.columns if col not in ["timestamp", "energy_kwh"]]

    split = int(len(df) * (1 - TEST_SIZE_RATIO))

    train = df[:split]
    test = df[split:]

    X_train = train[feature_cols]
    y_train = train["energy_kwh"]

    X_test = test[feature_cols]
    y_test = test["energy_kwh"]

    print("Training...")
    models = train_models(X_train, y_train)

    print("Selecting best model...")
    best_model = select_best(models, X_test, y_test)

    preds = best_model.predict(X_test)

    plot_results(y_test, preds, IMAGES_DIR / "result.png")

    save_model(best_model, MODEL_FILE)

    print("✅ Done!")

if __name__ == "__main__":
    main()
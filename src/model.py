import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_models(X_train, y_train):

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return {"lr": lr, "rf": rf}


def evaluate(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return rmse, r2


def select_best(models, X_test, y_test):

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():

        preds = model.predict(X_test)
        rmse, r2 = evaluate(y_test, preds)

        print(f"{name} → RMSE: {rmse:.2f}, R2: {r2:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    return best_model


def save_model(model, path):
    joblib.dump(model, path)
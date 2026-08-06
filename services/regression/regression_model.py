import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import PolynomialFeatures

from services.regression.preprocessing import preprocess_regression


def train_regression():


    data = preprocess_regression()

    X = data["X"]

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    scaler = data["scaler"]

 
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)


    poly = PolynomialFeatures(
        degree=2,
        include_bias=False,
    )

    X_train_poly = poly.fit_transform(
        X_train_scaled
    )

    X_test_poly = poly.transform(
        X_test_scaled
    )


    model = Ridge(alpha=100)

    model.fit(
        X_train_poly,
        y_train_log,
    )

  
    y_pred_log = model.predict(
        X_test_poly
    )

    y_pred = np.expm1(y_pred_log)

    y_actual = np.expm1(y_test_log)


    r2 = r2_score(
        y_actual,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_actual,
        y_pred,
    )

    results = pd.DataFrame({
        "Actual Price": y_actual,
        "Predicted Price": y_pred
    })

    return {
        "model": model,
        "poly": poly,
        "scaler": scaler,
        "features": X.columns,
        "results": results,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


def predict_price(
    model,
    scaler,
    poly,
    input_df,
):

    scaled = scaler.transform(
        input_df
    )

    poly_data = poly.transform(
        scaled
    )

    log_price = model.predict(
        poly_data
    )

    return np.expm1(log_price)[0]
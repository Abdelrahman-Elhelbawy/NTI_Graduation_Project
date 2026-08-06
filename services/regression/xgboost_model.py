import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from xgboost import XGBRegressor

from services.regression.preprocessing import preprocess_regression


def train_xgboost():

    # ====================================
    # Load Preprocessed Data
    # ====================================

    data = preprocess_regression()

    X = data["X"]

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    # ====================================
    # Log Transform
    # ====================================

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # ====================================
    # Model
    # ====================================

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(
        X_train,
        y_train_log
    )

    # ====================================
    # Prediction
    # ====================================

    y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)

    y_actual = np.expm1(y_test_log)

    # ====================================
    # Metrics
    # ====================================

    r2 = r2_score(
        y_actual,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_actual,
        y_pred
    )

    # ====================================
    # Prediction Table
    # ====================================

    results = pd.DataFrame({
        "Actual Price": y_actual,
        "Predicted Price": y_pred
    })

    # ====================================
    # Feature Importance
    # ====================================

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    return {
        "model": model,
        "features": X.columns,
        "results": results,
        "importance": importance,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


def predict_price(model, input_df):

    log_price = model.predict(input_df)

    return np.expm1(log_price)[0]
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from services.regression.preprocessing import preprocess_regression

def train_xgboost():
    data = preprocess_regression()
    X = data["X"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    y_train_log = np.log1p(y_train) if np.max(y_train) > 100 else y_train
    y_test_log = np.log1p(y_test) if np.max(y_test) > 100 else y_test

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test_log)

    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)

    results = pd.DataFrame({
        "Actual Price": y_actual,
        "Predicted Price": y_pred
    })

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return {
        "model": model,
        "features": X.columns,
        "results": results,
        "importance": importance,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }

def predict_price(model, features, input_df):
    clean_df = input_df.drop(columns=[col for col in ['id', 'date', 'price'] if col in input_df.columns])
    clean_df = clean_df[features]
    log_price = model.predict(clean_df)
    return np.expm1(log_price)[0]
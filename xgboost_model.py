import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def train_model():
    df = pd.read_csv('kc_house_data.csv')
    X = df.drop(columns=['id', 'date', 'price'])
    y = np.log1p(df['price'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return model, X.columns

def predict_price(model, input_df):
    log_price = model.predict(input_df)
    actual_price = np.expm1(log_price)
    return actual_price[0]

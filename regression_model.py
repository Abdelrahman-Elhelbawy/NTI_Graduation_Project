import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    df = pd.read_csv('kc_house_data.csv')
    X = df.drop(columns=['id', 'date', 'price'])
    y = np.log1p(df['price'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_poly, y_train)
    
    return model, scaler, poly, X.columns

def predict_price(model, scaler, poly, input_df):
    input_scaled = scaler.transform(input_df)
    input_poly = poly.transform(input_scaled)
    log_price = model.predict(input_poly)
    actual_price = np.expm1(log_price)
    return actual_price[0]

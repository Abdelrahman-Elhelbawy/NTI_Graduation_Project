import streamlit as st
import matplotlib.pyplot as plt

from services.regression.xgboost_model import (
    train_xgboost,
    predict_price,
)

st.set_page_config(
    page_title="XGBoost",
    layout="wide",
)

st.title("XGBoost Regression")


result = train_xgboost()


st.header("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "R² Score",
        f"{result['r2']:.4f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"{result['rmse']:.2f}"
    )

with col3:
    st.metric(
        "MAE",
        f"{result['mae']:.2f}"
    )

st.divider()

st.header("Feature Importance")

fig, ax = plt.subplots(figsize=(10,6))

importance = result["importance"].sort_values(
    by="Importance"
)

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_xlabel("Importance")

st.pyplot(fig)

st.divider()


st.header("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(
    result["results"]["Actual Price"],
    result["results"]["Predicted Price"],
    alpha=0.7
)

ax.plot(
    [
        result["results"]["Actual Price"].min(),
        result["results"]["Actual Price"].max()
    ],
    [
        result["results"]["Actual Price"].min(),
        result["results"]["Actual Price"].max()
    ],
    color="red"
)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")

st.pyplot(fig)

st.divider()


st.header("Residual Plot")

residuals = (
    result["results"]["Actual Price"]
    -
    result["results"]["Predicted Price"]
)

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(
    result["results"]["Predicted Price"],
    residuals,
    alpha=0.7
)

ax.axhline(
    0,
    color="red",
    linestyle="--"
)

ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residual")

st.pyplot(fig)

st.divider()


st.header("Predict House Price")

user_input = {}

for feature in result["features"]:

    user_input[feature] = st.number_input(
        feature,
        value=0.0,
        key=feature
    )

if st.button("Predict"):

    import pandas as pd

    input_df = pd.DataFrame(
        [user_input]
    )

    prediction = predict_price(
        result["model"],
        input_df
    )

    st.success(
        f"Predicted Price : ${prediction:,.2f}"
    )
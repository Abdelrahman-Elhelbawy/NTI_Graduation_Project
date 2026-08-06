import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from services.regression.regression_model import train_regression
from services.regression.xgboost_model import train_xgboost

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

st.title("Regression Models Comparison")


with st.spinner("Training Models..."):
    ridge = train_regression()
    xgb = train_xgboost()


st.header("Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ridge Regression")
    st.metric("R² Score", f"{ridge['r2']:.4f}")
    st.metric("RMSE", f"{ridge['rmse']:.2f}")
    st.metric("MAE", f"{ridge['mae']:.2f}")

with col2:
    st.subheader("XGBoost")
    st.metric("R² Score", f"{xgb['r2']:.4f}")
    st.metric("RMSE", f"{xgb['rmse']:.2f}")
    st.metric("MAE", f"{xgb['mae']:.2f}")

st.divider()


comparison_df = pd.DataFrame({
    "Metric": ["R² Score", "RMSE", "MAE"],
    "Ridge Regression": [
        ridge["r2"],
        ridge["rmse"],
        ridge["mae"]
    ],
    "XGBoost": [
        xgb["r2"],
        xgb["rmse"],
        xgb["mae"]
    ]
})

st.header("Comparison Table")

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

st.divider()


st.header("Metrics Comparison")

fig, ax = plt.subplots(figsize=(8,5))

metrics = ["R²", "RMSE", "MAE"]

ridge_values = [
    ridge["r2"],
    ridge["rmse"],
    ridge["mae"]
]

xgb_values = [
    xgb["r2"],
    xgb["rmse"],
    xgb["mae"]
]

x = range(len(metrics))
width = 0.35

ax.bar(
    [i - width/2 for i in x],
    ridge_values,
    width,
    label="Ridge"
)

ax.bar(
    [i + width/2 for i in x],
    xgb_values,
    width,
    label="XGBoost"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.legend()

st.pyplot(fig)

st.divider()


st.header("Actual vs Predicted")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(
        ridge["results"]["Actual Price"],
        ridge["results"]["Predicted Price"],
        alpha=.7
    )

    ax.plot(
        [
            ridge["results"]["Actual Price"].min(),
            ridge["results"]["Actual Price"].max()
        ],
        [
            ridge["results"]["Actual Price"].min(),
            ridge["results"]["Actual Price"].max()
        ],
        color="red"
    )

    ax.set_title("Ridge Regression")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(
        xgb["results"]["Actual Price"],
        xgb["results"]["Predicted Price"],
        alpha=.7
    )

    ax.plot(
        [
            xgb["results"]["Actual Price"].min(),
            xgb["results"]["Actual Price"].max()
        ],
        [
            xgb["results"]["Actual Price"].min(),
            xgb["results"]["Actual Price"].max()
        ],
        color="red"
    )

    ax.set_title("XGBoost")

    st.pyplot(fig)

st.divider()


st.header("Best Model")

if (
    xgb["r2"] > ridge["r2"]
    and xgb["rmse"] < ridge["rmse"]
    and xgb["mae"] < ridge["mae"]
):

    st.success(
        "XGBoost achieved the best overall performance."
    )

else:

    st.success(
        "Ridge Regression achieved the best overall performance."
    )
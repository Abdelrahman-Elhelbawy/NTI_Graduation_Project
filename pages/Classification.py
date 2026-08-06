import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from services.classification.classification import (
    train_xgboost_classifier,
    predict_price_category,
)

st.set_page_config(
    page_title="Classification",
    layout="wide"
)

st.title("XGBoost Classification")


with st.spinner("Training XGBoost Classifier..."):
    result = train_xgboost_classifier()


st.header("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{result['accuracy']:.4f}"
    )

with col2:
    st.metric(
        "Features",
        len(result["feature_names"])
    )

with col3:
    st.metric(
        "Classes",
        len(result["class_names"])
    )

st.divider()


st.header("Classification Report")

st.dataframe(
    result["report"],
    use_container_width=True
)

st.divider()


st.header("Feature Importance")

col1, col2 = st.columns([1, 2])

with col1:

    st.dataframe(
        result["importance"],
        use_container_width=True,
        hide_index=True
    )

with col2:

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=result["importance"],
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)

st.divider()

st.header("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    result["confusion"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=result["class_names"],
    yticklabels=result["class_names"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.divider()

st.header("Prediction Results")

st.dataframe(
    result["prediction_table"],
    use_container_width=True,
    hide_index=True
)

st.divider()


st.header("Top 10 Important Features")

st.dataframe(
    result["importance"].head(10),
    use_container_width=True,
    hide_index=True
)

st.divider()


st.divider()

st.header("Predict House Category")

with st.form("prediction_form"):

    user_input = {}

    columns = st.columns(3)

    for index, feature in enumerate(result["feature_names"]):

        with columns[index % 3]:

            user_input[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f"
            )

    submitted = st.form_submit_button("Predict")

if submitted:

    input_df = pd.DataFrame([user_input])

    prediction = predict_price_category(
        result["model"],
        input_df,
    )

    st.success(
        f"Predicted Category: {prediction}"
    )
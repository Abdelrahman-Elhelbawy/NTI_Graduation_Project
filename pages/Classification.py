import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from services.classification.classification import train_xgboost_classifier

st.set_page_config(
    page_title="Classification",
    layout="wide"
)

st.title("XGBoost Classification")

# =====================================================
# Train Model
# =====================================================

with st.spinner("Training XGBoost Classifier..."):
    result = train_xgboost_classifier()

# =====================================================
# Metrics
# =====================================================

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

# =====================================================
# Classification Report
# =====================================================

st.header("Classification Report")

st.dataframe(
    result["report"],
    use_container_width=True
)

st.divider()

# =====================================================
# Feature Importance
# =====================================================

st.header("Feature Importance")

col1, col2 = st.columns([1, 2])

with col1:

    st.dataframe(
        result["importance"],
        use_container_width=True,
        hide_index=True
    )

with col2:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=result["importance"],
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=ax
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")

    st.pyplot(fig)

st.divider()

# =====================================================
# Confusion Matrix
# =====================================================

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

# =====================================================
# Prediction Results
# =====================================================

st.header("Prediction Results")

st.dataframe(
    result["prediction_table"],
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# Top Important Features
# =====================================================

st.header("Top 10 Important Features")

st.dataframe(
    result["importance"].head(10),
    use_container_width=True,
    hide_index=True
)
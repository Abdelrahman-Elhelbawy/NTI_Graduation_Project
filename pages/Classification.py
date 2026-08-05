import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree

from services.classification.classification import train_decision_tree


st.set_page_config(
    page_title="Classification",
    layout="wide"
)

st.title("Decision Tree Classification")

# ==========================================
# Train Model
# ==========================================

with st.spinner("Training Decision Tree Model..."):
    result = train_decision_tree()

# ==========================================
# Metrics
# ==========================================

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{result['accuracy']:.4f}"
    )

with col2:
    st.metric(
        "Number of Features",
        len(result["feature_names"])
    )

st.divider()

# ==========================================
# Classification Report
# ==========================================

st.header("Classification Report")

st.dataframe(
    result["report"],
    use_container_width=True
)

st.divider()

# ==========================================
# Feature Importance
# ==========================================

st.header("Feature Importance")

col1, col2 = st.columns([1, 2])

with col1:

    st.dataframe(
        result["importance"],
        use_container_width=True
    )

with col2:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=result["importance"],
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)

st.divider()

# ==========================================
# Confusion Matrix
# ==========================================

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

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

st.pyplot(fig)

st.divider()

# ==========================================
# Decision Tree
# ==========================================

st.header("Decision Tree")

fig, ax = plt.subplots(figsize=(24, 12))

plot_tree(
    result["model"],
    feature_names=result["feature_names"],
    class_names=result["class_names"],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=8,
    ax=ax
)

st.pyplot(fig)

st.divider()

# ==========================================
# Feature Importance Table
# ==========================================

st.header("Feature Ranking")

st.dataframe(
    result["importance"],
    use_container_width=True
)
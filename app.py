import streamlit as st

st.set_page_config(
    page_title="NTI Graduation Project",
    layout="wide",
)

# ==========================
# Pages
# ==========================

home = st.Page(
    "pages/home.py",
    title="Home",
    # icon="🏠"
)

cleaning = st.Page(
    "pages/Data_Cleaning.py",
    title="Data Cleaning",
    # icon="🧹"
)

visualization = st.Page(
    "pages/Data_Visualization.py",
    title="Data Visualization",
    # icon="📊"
)

comparison = st.Page(
    "pages/Model_Comparison.py",
    title="Model Comparison"
)


classification = st.Page(
    "pages/Classification.py",
    title="Classification",
    # icon="🌳"
)

regression = st.Page(
    "pages/Regression.py",
    title="Regression",
    # icon="📈"
)

xgboost = st.Page(
    "pages/XGBoost.py",
    title="XGBoost",
    # icon="⚡"
)

contributors = st.Page(
    "pages/Contributors.py",
    title="Contributors",
    # icon="👥"
)

# ==========================
# Navigation
# ==========================

pg = st.navigation(
    {
        "NTI Graduation Project": [
            home,
            cleaning,
            visualization,
            classification,
            regression,
            xgboost,
            comparison,
        ],
        "About": [
            contributors,
        ],
    }
)

pg.run()
import streamlit as st

from Data_Cleaning import clean_data

st.set_page_config(
    page_title="Data Cleaning",
    layout="wide"
)

st.title("🧹 Data Cleaning")

result = clean_data()

# =====================================
# Dataset Overview
# =====================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", result["shape_before"][0])
col2.metric("Columns", result["shape_before"][1])
col3.metric("Duplicates", result["duplicates"])

st.divider()

# =====================================
# Preview
# =====================================

tab1, tab2 = st.tabs(["Head", "Tail"])

with tab1:
    st.dataframe(result["head"], use_container_width=True)

with tab2:
    st.dataframe(result["tail"], use_container_width=True)

# =====================================
# Data Types
# =====================================

st.header("Data Types")

st.dataframe(
    result["dtypes"],
    use_container_width=True
)

# =====================================
# Missing Values
# =====================================

st.header("Missing Values")

st.dataframe(
    result["missing"],
    use_container_width=True
)

# =====================================
# Statistics
# =====================================

st.header("Statistical Summary")

st.dataframe(
    result["describe"],
    use_container_width=True
)

# =====================================
# Outlier Report
# =====================================

st.header("Outlier Treatment")

st.dataframe(
    result["outliers"],
    use_container_width=True
)

# =====================================
# Final Dataset
# =====================================

st.header("Cleaned Dataset")

st.success(
    f"Shape After Cleaning : {result['shape_after'][0]} Rows × {result['shape_after'][1]} Columns"
)

st.dataframe(
    result["df"],
    use_container_width=True
)
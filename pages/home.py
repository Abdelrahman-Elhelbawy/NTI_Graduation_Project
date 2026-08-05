import streamlit as st

st.set_page_config(
    page_title="NTI Graduation Project",
    layout="wide",
)

st.title("NTI Graduation Project")

st.write(
    """
Welcome to the **NTI Graduation Project Dashboard**.

This application provides an interactive environment for data analysis
and machine learning tasks.
"""
)

st.divider()

st.subheader("Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- Upload Datasets
- Data Cleaning
- Exploratory Data Analysis (EDA)
""")

with col2:
    st.markdown("""
- Data Visualization
- Machine Learning Models
- Predictions & Results
""")

st.divider()

st.info("Use the sidebar to navigate between the application pages.")
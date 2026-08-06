import streamlit as st

st.set_page_config(
    page_title="Contributors",
    layout="wide"
)

st.title("Project Contributors")

st.markdown(
    """
    This project was developed by the following team members.
    """
)

st.divider()

contributors = [
    "Abdelrahman Elhelbawy",
    "Abanob Nabil",
    "Marial Michel",
    "Mariam Ali",
    "Eman Ahmed",
    "Yomna Emad",
]

cols = st.columns(3)

for index, name in enumerate(contributors):
    with cols[index % 3]:
        with st.container(border=True):
            st.subheader(name)
            st.caption("Team Member")

st.divider()

st.markdown(
    """
    **Project:** House Price Prediction

    **Platform:** Streamlit

    **Organization:** NTI Graduation Project
    """
)
import streamlit as st

st.set_page_config(
    page_title="Contributors",
    layout="wide"
)

st.title("Contributors")

st.write("Meet the project team.")

st.divider()

contributors = [
    "Abdelrahman Elhelbawy",
    "Abanob Nabil",
    "Marial Michel",
    "Mariam Ali",
    "Eman Ahmed",
    "Yomna Emad",
]

cols = st.columns(2)

for i, name in enumerate(contributors):
    with cols[i % 2]:
        st.container(border=True)
        st.subheader(name)
        st.caption("Team Member")
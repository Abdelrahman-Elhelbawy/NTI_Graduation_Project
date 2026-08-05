import streamlit as st

home = st.Page("pages/home.py", title="Home")
cleaning = st.Page("pages/Data_cleaning.py", title="Data Cleaning")

pg = st.navigation({
    "NTI Graduation Project": [
        home,
        cleaning,
    ]
})

pg.run()
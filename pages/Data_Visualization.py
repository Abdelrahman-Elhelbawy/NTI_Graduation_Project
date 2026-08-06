import streamlit as st

from services.Data_visualization import *

st.set_page_config(
    page_title="Data Visualization",
    layout="wide"
)

st.title("Data Visualization")


@st.cache_data
def get_data():
    return load_data()


df = get_data()


st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.divider()


st.header("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.divider()


st.header("Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        price_distribution(df)
    )

with col2:
    st.pyplot(
        price_boxplot(df)
    )

st.divider()


st.header("Correlation Analysis")

st.pyplot(
    correlation_heatmap(df)
)

st.pyplot(
    correlation_with_price(df)
)

st.divider()


st.header("Living Area Analysis")

st.pyplot(
    living_area_vs_price(df)
)

st.divider()


st.header("House Grade")

st.pyplot(
    price_by_grade(df)
)

st.divider()


st.header("Waterfront")

st.pyplot(
    price_by_waterfront(df)
)

st.divider()


st.header("House Condition")

st.pyplot(
    price_by_condition(df)
)

st.divider()


st.header("House View")

st.pyplot(
    price_by_view(df)
)

st.divider()


st.header("Bedrooms Analysis")

col1, col2 = st.columns(2)

with col1:

    st.pyplot(
        average_price_by_bedrooms(df)
    )

with col2:

    st.pyplot(
        houses_by_bedrooms(df)
    )

st.divider()


st.header("House Locations")

st.pyplot(
    house_locations(df)
)

st.divider()


st.header("Pair Plot")

st.pyplot(
    pair_plot(df)
)
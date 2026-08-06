import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    font_scale=1.1
)


def load_data():

    return pd.read_csv(
        "data/kc_house_cleaned.csv"
    )



def price_distribution(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        df["price"],
        bins=30,
        kde=True,
        color="royalblue",
        ax=ax
    )

    ax.set_title("Distribution of House Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")

    return fig



def price_boxplot(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        y=df["price"],
        color="skyblue",
        ax=ax
    )

    ax.set_title("Box Plot of House Prices")

    return fig



def correlation_heatmap(df):

    fig, ax = plt.subplots(figsize=(14, 10))

    corr = df.corr(
        numeric_only=True
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    return fig



def correlation_with_price(df):

    fig, ax = plt.subplots(figsize=(8, 10))

    corr = (
        df
        .corr(numeric_only=True)["price"]
        .sort_values()
    )

    corr.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Correlation with Price")

    return fig



def living_area_vs_price(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="sqft_living",
        y="price",
        alpha=.6,
        ax=ax
    )

    ax.set_title("Living Area vs Price")

    return fig



def price_by_grade(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="grade",
        y="price",
        palette="Blues",
        ax=ax
    )

    ax.set_title("House Price by Grade")

    return fig



def price_by_waterfront(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="waterfront",
        y="price",
        palette="Set2",
        ax=ax
    )

    ax.set_title("House Price by Waterfront")

    return fig



def price_by_condition(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="condition",
        y="price",
        ax=ax
    )

    ax.set_title("House Price by Condition")

    return fig



def price_by_view(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="view",
        y="price",
        ax=ax
    )

    ax.set_title("House Price by View")

    return fig



def average_price_by_bedrooms(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="bedrooms",
        y="price",
        ax=ax
    )

    ax.set_title("Average Price by Bedrooms")

    return fig



def houses_by_bedrooms(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="bedrooms",
        ax=ax
    )

    ax.set_title("Number of Houses by Bedrooms")

    return fig



def house_locations(df):

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x="long",
        y="lat",
        hue="price",
        palette="viridis",
        alpha=.6,
        ax=ax
    )

    ax.set_title("House Locations")

    return fig



def pair_plot(df):

    fig = sns.pairplot(

        df[
            [
                "price",
                "sqft_living",
                "sqft_lot",
                "grade",
                "bathrooms"
            ]
        ]

    )

    return fig.fig
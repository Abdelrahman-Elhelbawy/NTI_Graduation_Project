import pandas as pd
import numpy as np


def clean_data():

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv("data/kc_house_data.csv")

    # -----------------------------
    # Basic Information
    # -----------------------------
    head = df.head()
    tail = df.tail()
    shape_before = df.shape
    describe = df.describe()

    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["Column", "Data Type"]

    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]

    duplicate_count = df.duplicated().sum()

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")

    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month

    df["was_renovated"] = (df["yr_renovated"] != 0).astype(int)

    # -----------------------------
    # Outlier Handling
    # -----------------------------
    columns_to_cap = [
        "bedrooms",
        "bathrooms",
        "sqft_lot",
        "sqft_lot15",
    ]

    outlier_report = []

    for col in columns_to_cap:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        low_values = (df[col] < lower).sum()
        high_values = (df[col] > upper).sum()

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

        outlier_report.append({
            "Column": col,
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Low Values": low_values,
            "High Values": high_values
        })

    # -----------------------------
    # Drop Columns
    # -----------------------------
    df = df.drop(columns=["id", "date", "yr_renovated"])

    shape_after = df.shape

    print("_______________________________________________________________")
    print("Shape after cleaning:", shape_after)

    # -----------------------------
    # Save Cleaned Dataset
    # -----------------------------
    df.to_csv("data/kc_house_cleaned.csv", index=False)

    print("Cleaned dataset saved successfully!")

    # -----------------------------
    # Return Results
    # -----------------------------
    return {
        "df": df,
        "head": head,
        "tail": tail,
        "shape_before": shape_before,
        "shape_after": shape_after,
        "describe": describe,
        "dtypes": dtypes,
        "missing": missing_values,
        "duplicates": duplicate_count,
        "outliers": pd.DataFrame(outlier_report)
    }
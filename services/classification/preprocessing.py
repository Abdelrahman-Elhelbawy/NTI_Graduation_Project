import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_classification():

    # ===========================
    # Load Dataset
    # ===========================

    df = pd.read_csv("data/kc_house_cleaned.csv")

    # ===========================
    # Create Target
    # ===========================

    df["Price_Category"] = df["price"].apply(
        lambda x:
        "Low"
        if x < 350000
        else "Medium"
        if x <= 650000
        else "Luxury"
    )

    # ===========================
    # Encode Target
    # ===========================

    le = LabelEncoder()

    df["Price_Category"] = le.fit_transform(
        df["Price_Category"]
    )

    # ===========================
    # Features & Target
    # ===========================

    X = df.drop(
        ["price", "Price_Category"],
        axis=1
    )

    y = df["Price_Category"]

    # ===========================
    # Train Test Split
    # ===========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": le,
    }
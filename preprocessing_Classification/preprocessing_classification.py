# code

# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load Cleaned Dataset


df = pd.read_csv("data/kc_house_cleaned.csv")


# Create Price Category


df["Price_Category"] = df["price"].apply(
    lambda x: "Low"
    if x < 350000
    else "Medium"
    if x <= 650000
    else "Luxury"
)


# Encode Target


le = LabelEncoder()

df["Price_Category"] = le.fit_transform(df["Price_Category"])


# Feature Selection


X = df.drop(["price", "Price_Category"], axis=1)
y = df["Price_Category"]


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)





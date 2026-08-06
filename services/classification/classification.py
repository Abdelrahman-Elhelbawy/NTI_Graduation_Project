import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


def train_xgboost_classifier():
    """
    Train an XGBoost classifier for House Price Category prediction.
    """

    # ==========================================
    # Load Dataset
    # ==========================================

    df = pd.read_csv("data/kc_house_cleaned.csv")

    # ==========================================
    # Feature Engineering
    # ==========================================

    df["house_age"] = df["sale_year"] - df["yr_built"]

    # ==========================================
    # Create Target
    # ==========================================

    df["Price_Category"] = df["price"].apply(
        lambda x: "Low" if x < 450000 else "High"
    )

    # ==========================================
    # Encode Target
    # ==========================================

    le = LabelEncoder()

    df["Price_Category"] = le.fit_transform(
        df["Price_Category"]
    )

    # ==========================================
    # Features & Target
    # ==========================================

    X = df.drop(
        ["price", "Price_Category"],
        axis=1
    )

    y = df["Price_Category"]

    # ==========================================
    # Train Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ==========================================
    # Model
    # ==========================================

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    # ==========================================
    # Prediction
    # ==========================================

    y_pred = model.predict(
        X_test
    )

    # ==========================================
    # Metrics
    # ==========================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=le.classes_,
            output_dict=True,
        )
    ).transpose()

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        by="Importance",
        ascending=False,
    )

    prediction_table = pd.DataFrame(
        {
            "Actual": le.inverse_transform(y_test),
            "Predicted": le.inverse_transform(y_pred),
        }
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "confusion": confusion,
        "importance": importance,
        "prediction_table": prediction_table,
        "feature_names": X.columns,
        "class_names": le.classes_,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_price_category(model, input_df):
    """
    Predict house price category.
    """

    prediction = model.predict(input_df)

    if prediction[0] == 0:
        return "Low"

    return "High"
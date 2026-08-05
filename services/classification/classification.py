import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from services.classification.preprocessing import preprocess_classification

data = preprocess_classification()

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

X = data["X"]

le = data["label_encoder"]


def train_decision_tree():

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        criterion="entropy",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=le.classes_,
            output_dict=True,
        )
    ).transpose()

    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    confusion = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "importance": importance,
        "confusion": confusion,
        "feature_names": X.columns,
        "class_names": le.classes_,
    }
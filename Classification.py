import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------
# preprocessing_Classification
#-----------------------------
# -----------------------------
# Load Cleaned Dataset
# -----------------------------
df = pd.read_csv("data/kc_house_cleaned.csv")


# -----------------------------
# Price Category: فئتين (Low/High) بحد فاصل = median السعر
# -----------------------------
df["Price_Category"] = df["price"].apply(
    lambda x: "Low" if x < 450000 else "High"
)

# -----------------------------
# Encode Target
# -----------------------------
le = LabelEncoder()
df["Price_Category"] = le.fit_transform(df["Price_Category"])

# -----------------------------
# Feature Selection
# -----------------------------
X = df.drop(["price", "Price_Category"], axis=1)
y = df["Price_Category"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#-----------------------------
# preprocessing_Classification
#-----------------------------



# Feature: عمر البيت وقت البيع
df["house_age"] = df["sale_year"] - df["yr_built"]

# XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Evaluation
y_pred = xgb_model.predict(X_test)

print("--- XGBoost Results (Low/High) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, target_names=le.classes_),
)

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.title("Confusion Matrix - XGBoost (Low/High)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importance:\n", importance)


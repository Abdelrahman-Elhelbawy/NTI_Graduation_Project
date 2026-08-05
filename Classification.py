import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn. tree import plot_tree


from preprocessing_Classification.preprocessing_classification import X_train, X_test, y_train, y_test, le, X


dt_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    criterion='entropy',
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_dt))
print("Classification Report: ", classification_report(y_test, y_pred_dt, target_names=le.classes_))


importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
})
importance = importance.sort_values('Importance',ascending=False)
print(importance)

plt.figure(figsize=(12,8))
plot_tree(
dt_model,
feature_names=X.columns,
class_names=le.classes_,
filled=True,
rounded=True,
impurity=False,
fontsize=10
)
plt.title("Decision Tree Classifier")
plt.show()



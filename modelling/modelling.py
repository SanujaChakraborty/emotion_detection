import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Load features
X_train = pd.read_csv("data/processed/train_features.csv")
X_test = pd.read_csv("data/processed/test_features.csv")

# Load labels
y_train = pd.read_csv("data/processed/train_labels.csv").values.ravel()
y_test = pd.read_csv("data/processed/test_labels.csv").values.ravel()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic_regression_model.pkl")

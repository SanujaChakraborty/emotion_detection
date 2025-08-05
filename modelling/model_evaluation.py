# model_evaluation.py
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json

# Load the vectorizer used during training
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load test data and transform using SAME vectorizer
test_df = pd.read_csv("data/processed/test.csv")  # or wherever your clean test data is
test_df.dropna(subset=['content'], inplace=True)
X_test = vectorizer.transform(test_df['content'])
y_test = test_df['sentiment'].values

# Load model
model = pickle.load(open("models/random_forest_model.pkl", "rb"))

# Predict and compute metrics
y_pred = model.predict(X_test)

metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

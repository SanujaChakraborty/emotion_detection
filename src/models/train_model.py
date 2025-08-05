import pandas as pd
import pickle
import os
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Load hyperparameters from params.yaml ===
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_estimators = params["model_building"]["n_estimators"]
max_depth = params["model_building"]["max_depth"]

# === Load features ===
with open("data/features/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("data/features/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

# === Load labels ===
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
y_train = train_df["sentiment"]
y_test = test_df["sentiment"]

# === Train the model ===
model = RandomForestClassifier(
    n_estimators=n_estimators, 
    max_depth=max_depth, 
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")

# === Save model ===
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# === Save metrics (optional, for DVC metrics tracking) ===
os.makedirs("metrics", exist_ok=True)
with open("metrics/accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")

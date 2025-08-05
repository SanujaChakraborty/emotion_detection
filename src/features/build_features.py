import pandas as pd
import os
import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_features(train_df, test_df, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['content'])
    X_test = vectorizer.transform(test_df['content'])

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

    # Extract labels separately
    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    return X_train_df, X_test_df, y_train, y_test, vectorizer

if __name__ == "__main__":
    params = load_params()
    max_features = params["build_features"]["max_features"]

    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    X_train_df, X_test_df, y_train, y_test, vectorizer = build_features(train_df, test_df, max_features)

    os.makedirs("data/processed", exist_ok=True)

    # Save features
    X_train_df.to_csv("data/processed/train_features.csv", index=False)
    X_test_df.to_csv("data/processed/test_features.csv", index=False)

    # Save labels separately
    y_train.to_csv("data/processed/train_labels.csv", index=False)
    y_test.to_csv("data/processed/test_labels.csv", index=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.joblib")

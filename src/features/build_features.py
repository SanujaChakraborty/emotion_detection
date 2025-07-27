import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load processed training and test data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# ✅ Drop NaN rows (VERY IMPORTANT)
train_df.dropna(subset=['content'], inplace=True)
test_df.dropna(subset=['content'], inplace=True)


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform training content
X_train = vectorizer.fit_transform(train_df['content'])

# Transform test content
X_test = vectorizer.transform(test_df['content'])

# Save feature matrices
os.makedirs("data/features", exist_ok=True)
with open("data/features/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("data/features/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

# Save vectorizer itself
with open("data/features/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

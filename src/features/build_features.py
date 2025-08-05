import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import yaml

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

max_features = params["feature_engg"]["max_features"]

# Load processed training and test data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Drop NaN rows (VERY IMPORTANT)
train_df.dropna(subset=['content'], inplace=True)
test_df.dropna(subset=['content'], inplace=True)

# Assuming your train_df and test_df are already loaded
vectorizer = TfidfVectorizer(max_features=4999)
X_train = vectorizer.fit_transform(train_df['content'])
X_test = vectorizer.transform(test_df['content'])

os.makedirs("data/features", exist_ok=True)

# Save X_train and X_test as .pkl
with open("data/features/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("data/features/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)


# Save BOW CSVs
df_train_bow = pd.DataFrame(X_train.toarray())
df_train_bow['label'] = train_df['sentiment'].values
df_train_bow.to_csv("data/interim/train_bow.csv", index=False)

df_test_bow = pd.DataFrame(X_test.toarray())
df_test_bow['label'] = test_df['sentiment'].values
df_test_bow.to_csv("data/interim/test_bow.csv", index=False)

with open("data/features/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

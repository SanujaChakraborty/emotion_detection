# src/data/data_ingestion.py

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from typing import Tuple

# Setup logger
logging.basicConfig(
    filename='logs/data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded from YAML successfully.")
        return params
    except Exception as e:
        logging.error(f"Error reading params.yaml: {e}")
        raise

def load_dataset(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully from URL.")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["tweet_id"], inplace=True)
        df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        df["sentiment"] = df["sentiment"].map({"happiness": 1, "sadness": 0})
        logging.info("Dataset preprocessing completed.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_and_save_data(
    df: pd.DataFrame, 
    test_size: float, 
    train_path: str, 
    test_path: str
) -> None:
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train/Test data saved to {train_path} and {test_path}.")
    except Exception as e:
        logging.error(f"Error splitting/saving data: {e}")
        raise

if __name__ == "__main__":
    params = load_params("params.yaml")
    raw_df = load_dataset(params["data_ingestion"]["source_url"])
    processed_df = preprocess_dataset(raw_df)
    split_and_save_data(
        processed_df,
        test_size=params["data_ingestion"]["test_size"],
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv"
    )

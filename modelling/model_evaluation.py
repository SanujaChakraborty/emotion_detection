import pandas as pd
import joblib
import logging
from sklearn.metrics import classification_report, accuracy_score

# Setup logging
logging.basicConfig(filename='evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(test_features_path: str, test_labels_path: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = pd.read_csv(test_features_path)
        y_test = pd.read_csv(test_labels_path)
        logging.info("Test data loaded successfully.")
        return X_test, y_test.squeeze()
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    try:
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logging.info(f"Model accuracy: {acc}")
        print("Classification Report:\n", report)
        print("Accuracy Score:", acc)
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    X_test, y_test = load_data("data/processed/test_features.csv", "data/processed/test_labels.csv")
    model = load_model("models/logistic_regression_model.pkl")
    evaluate_model(model, X_test, y_test)

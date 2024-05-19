from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from pandas import DataFrame
from pathlib import Path
from typing import Dict
import numpy as np
import logging
import sys
import yaml

logger = logging.getLogger(__name__)

def evaluate_model(model: RandomForestClassifier, X_test: DataFrame, y_test: DataFrame) -> float:
    """
    Evaluate the model using accuracy score.

    Parameters:
    - model (BaseEstimator): The trained model to evaluate.
    - X_test (DataFrame): Testing features.
    - y_test (DataFrame): Testing target.

    Returns:
    - Dict[str, Any]: A dictionary containing the evaluation metrics.
    """
    
    logger.info("Predictiong values for test set")
    try:
        # Get model predictions for test set
        predictions = model.predict(X_test)
    except (KeyError, TypeError) as err:
        logger.error("An error occurred when predicting values for the test set. " +
                    "The process can't continue. Error: %s", err)
        sys.exit(1)
    except Exception as err:
        logger.error("An unexpected error occurred when predicting values for the test " +
                    "set. The process can't continue. Error: %s", err)
        sys.exit(1)
    else:
        logger.info("Predicted values for test set completed.")
    

    # Evaluate model
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }
    return metrics

def save_metrics(metrics_dict: dict, save_path: Path) -> None:
    """
    Save evaluation metrics to a YAML file. 

    Args:
        metrics_dict: A dictionary containing the evaluation metrics (AUC, confMatrix, Accuracy, 
                      classifReport)
        save_path: The local path to the file where to save metrics to 
    """

    # Convert numpy objects to Python types
    clean_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, np.ndarray):
            # Convert arrays to lists
            clean_metrics[key] = value.tolist()
        elif isinstance(value, np.generic):
            # Convert numpy scalars to Python scalars
            clean_metrics[key] = value.item()
        else:
            clean_metrics[key] = value

    # Save cleaned metrics into YAML file
    try:
        with open(save_path, "w", encoding="utf-8") as file:
            yaml.dump(clean_metrics, file, default_flow_style=False, sort_keys=False)
    except yaml.YAMLError as err:
        logger.warning("Failed to save metrics to YAML file %s. The process will continue " +
                       "without saving evaluation metrics. Error: %s", save_path, err)
    except Exception as err:
        logger.warning("An unexpected error occurred when saving metrics to %s. The process will " +
                       "continue without saving evaluation metrics. Error: %s", save_path, err)
    else:
        logger.info("Evaluation metrics dictionary successfully saved to %s", save_path)
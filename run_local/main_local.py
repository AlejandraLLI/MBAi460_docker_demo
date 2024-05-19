import pandas as pd
import logging.config
import yaml

import src.data_loader as dl
import src.model_training as mt
import src.evaluate_model as em
import os 
import datetime
from pathlib import Path

# Set logger
logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("docker_demo")


def lambda_handler(event, context):
    """
    Main function to run the application.
    """
    # Set logger
    logger.info("** APPLICATION STARTED **")

    # --- Read config file ---
    logger.info("Loading model configuration from config file.")
    with open("config/config.yaml", "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error loading model configuration from config/config.yaml file")
        else:
            logger.info("Model configuration file loaded from config/config.yaml file")

    # Set up output directory for saving artifacts, takes current timestamp as subfolder
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(config.get("run_config",{}).get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # --- Load data set ---
    logger.info("Loading data")
    data = dl.load_data(**config["load_data"])
    logger.info("Data loaded successfully")

    # --- Split data in train and test ---
    logger.info("Splitting data in train and test sets")
    X_train, X_test, y_train, y_test = dl.prepare_data(data, **config["prepare_data"])
    logger.info("Data successfully splited in test and train.")

    # --- Model training ---
    logger.info("Starting model training ... ")
    best_model, best_params = mt.create_and_tune_model(X_train, y_train, **config["training"])
    logger.info("Finished model training.")
    mt.save_hyperparams(best_params, artifacts / "best_params.csv")
    logger.info("Best hyperparameters saved to %s", artifacts / "best_params.csv")
    
    # ---- Model Evaluation --- 
    logger.info("Starting model scoring")
    metrics = em.evaluate_model(best_model, X_test, y_test)
    logger.info("Model scoring successfull.")
    logger.info("Evaluation Metrics:")
    logger.info(f'Accuracy: {metrics["accuracy"]:.2f}')
    logger.info(f'Precision: {metrics["precision"]:.2f}')
    logger.info(f'Recall: {metrics["recall"]:.2f}')
    logger.info(f'F1 Score: {metrics["f1_score"]:.2f}')
    logger.info(f'Confusion Matrix:\n{metrics["confusion_matrix"]}')
    em.save_metrics(metrics, artifacts / "model_eval_metrics.csv")
    logger.info("Model evaluation metrics saved to %s", artifacts / "model_eval_metrics.csv")


if __name__ == "__main__":
    # runn locally for testing 
    lambda_handler([],[])
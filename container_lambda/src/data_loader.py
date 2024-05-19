import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import logging 

logger = logging.getLogger(__name__)

def load_data(url: str, columns: List) -> pd.DataFrame:
    """
    Load data from a CSV file URL.

    Parameters:
    - url (str): URL of the CSV file.
    - columns (List): columns selected for training, including target variable

    Returns:
    - pd.DataFrame: Loaded data as a DataFrame.
    """
    logger.debug("URL: %s", url)
    logger.debug("Columns used: %s", columns)

    logger.info("Loading data from CSV in provided URL")
    # loading data from csv 
    data = pd.read_csv(url, names=columns)
    logger.info("Data loaded successfully. ncols: %s, nrows: %s", data.shape[0], data.shape[1])

    # Function output
    return data


def prepare_data(data: pd.DataFrame, target: str, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training by splitting into features and target, and then into training and testing sets.

    Parameters:
    - data (pd.DataFrame): The data to split.
    - target (str): name of target variable

    Returns:
    - Tuple containing training features (pd.DataFrame), testing features (pd.DataFrame), training target (pd.Series), and testing target (pd.Series).
    """

    # Select independent variables
    X = data.drop(target, axis=1)
    logger.info("Independent variables selected.")
    logger.debug("X dim: %s", X.shape)
    
    # Select dependent variable
    y = data[target]
    logger.info("Dependent variable extracted")

    # Split sample in train and test
    logger.info("Splitting data in train & test.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed)
    logger.info("Data split successfully.")
    logger.debug("X train shape: %s", X_train.shape)
    logger.debug("y train shape: %s", y_train.shape)
    logger.debug("X test shape: %s", X_test.shape)
    logger.debug("y yest shape: %s", y_test.shape)


    return X_train, X_test, y_train, y_test
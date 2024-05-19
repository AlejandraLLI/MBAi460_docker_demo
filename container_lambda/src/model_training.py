from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import logging 
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

def create_and_tune_model(X_train: DataFrame, y_train: DataFrame, param_grid: dict, cv_k: int, seed: int) -> RandomForestClassifier:
    """
    Create and tune a Random Forest model using GridSearchCV.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (DataFrame): Training target.
    - param_grid (dict): hyperparameter values to be used in gridsearch
    - cv_k (int): number of folds for cv
    - seed (int): seed to use for replicability

    Returns:
    - Tuple[BaseEstimator, dict]: The best estimator found by GridSearchCV and the best hyperparameters.
    """

    # select model
    model = RandomForestClassifier(random_state=seed)
    logger.info("Random Forest Classifier object created.")
    

    # Set gridsearch with CV to finetune hyperparams. 
    # NOTE: n_jobs = -1 sets all available CPU cores
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grid, 
                               cv=cv_k, 
                               n_jobs=-1)
    logger.info("Grid serc with %s fold CV set.", cv_k)
    
    # Execute grid search for train dataset 
    logger.info("Starting hyperparameter tuning ...")
    grid_search.fit(X_train, y_train)
    logger.info("Finished hyperparameter tuning.")

    # Get best hyperparams
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info("Best hyperparams: %s", best_params) 
    

    # Return best set of hyperparams.
    return best_estimator, best_params

def save_hyperparams(best_params: Dict, save_path: Path) -> None:
    """
    Save best hyperparams to a CSV file.

    Args:
        scores (pd.DataFrame): Pandas DataFrame with test scores.
        save_path (Path): Path to file where scores will be saved.
    """
    try:
        logger.info("Saving scores to file %s", save_path)
        # Convert dictionary to DataFrame
        df_hyperparams = DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
        
        # Save df
        df_hyperparams.to_csv(save_path, index = False)
    except FileNotFoundError:
        logger.warning("File %s not found. The process will continue without saving the scores " +
                       "to csv. Please provide a valid directory to save scores to.", save_path)
    except PermissionError:
        logger.warning("The process does not have the necessary permissions to create or write " +
                       "to the file %s. The process will continue without saving the scores.",
                        save_path)
    except Exception as err:
        logger.warning("An unexpected error occurred when saving scores to file %s. The process " +
                       "will continue without saving the scores. Error: %s", save_path, err)
    else:
        logger.info("Scores dataframed saved to file %s", save_path)
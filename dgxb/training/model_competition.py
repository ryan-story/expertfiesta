"""
Model competition: Train and optimize multiple models with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
import logging
import time
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_jobs: int = -1,
) -> Tuple[LinearRegression, Dict[str, Any], Dict[str, float], float]:
    """
    Train LinearRegression with hyperparameter optimization

    Args:
        X_train: Training features
        y_train: Training target (incident counts)
        cv_splits: List of (train_indices, test_indices) tuples
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, cv_scores, train_time_sec)
    """
    logger.info("Training LinearRegression...")

    # Hyperparameter grid
    param_grid = {
        "fit_intercept": [True, False],
    }

    # Create custom CV from splits
    cv = [(train_idx, test_idx) for (train_idx, test_idx) in cv_splits]

    # Create model with StandardScaler pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    # Update param grid for pipeline
    param_grid_pipeline = {
        "regressor__fit_intercept": param_grid["fit_intercept"],
    }

    # Use negative RMSE as scorer (GridSearchCV maximizes, so minimize RMSE = maximize -RMSE)
    scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )

    start_time = time.time()
    grid_search = GridSearchCV(
        pipeline,
        param_grid_pipeline,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to RMSE

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV score (RMSE): {best_score:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_model, best_params, {"rmse": best_score}, train_time


def train_poisson_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_jobs: int = -1,
) -> Tuple[PoissonRegressor, Dict[str, Any], Dict[str, float], float]:
    """
    Train PoissonRegressor (count-native model for incident counts)

    Args:
        X_train: Training features
        y_train: Training target (incident counts)
        cv_splits: List of (train_indices, test_indices) tuples
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, cv_scores, train_time_sec)
    """
    logger.info("Training PoissonRegressor...")

    # Hyperparameter grid
    param_grid = {
        "alpha": [0.0, 0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
    }

    # Create custom CV from splits
    cv = [(train_idx, test_idx) for (train_idx, test_idx) in cv_splits]

    # Create model with StandardScaler pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("regressor", PoissonRegressor())]
    )

    # Update param grid for pipeline
    param_grid_pipeline = {
        "regressor__alpha": param_grid["alpha"],
        "regressor__fit_intercept": param_grid["fit_intercept"],
    }

    # Use negative RMSE as scorer
    scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )

    start_time = time.time()
    grid_search = GridSearchCV(
        pipeline,
        param_grid_pipeline,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to RMSE

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV score (RMSE): {best_score:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_model, best_params, {"rmse": best_score}, train_time


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    Train LogisticRegression with L1 (Lasso) penalty and hyperparameter optimization
    DEPRECATED: Use train_linear_regression for regression tasks
    This function is kept for backward compatibility but should not be used.
    """
    raise NotImplementedError(
        "train_logistic_regression is deprecated. Use train_linear_regression for regression tasks."
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
    n_jobs: int = -1,
) -> Tuple[RandomForestRegressor, Dict[str, Any], Dict[str, float], float]:
    """
    Train RandomForestRegressor with hyperparameter optimization

    Args:
        X_train: Training features
        y_train: Training target (incident counts)
        cv_splits: List of (train_indices, test_indices) tuples
        n_trials: Number of random search trials
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, cv_scores, train_time_sec)
    """
    logger.info("Training RandomForest...")

    # Hyperparameter distribution
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    # Create custom CV from splits
    # Use list-of-tuples directly (PredefinedSplit causes leakage in rolling-origin CV)
    cv = [(train_idx, test_idx) for (train_idx, test_idx) in cv_splits]

    # Create model
    model = RandomForestRegressor(random_state=42, n_jobs=1)

    # Use negative RMSE as scorer
    scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )

    # Random search
    start_time = time.time()
    random_search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_trials,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
    )

    random_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    cv_scores = {
        "mean_test_score": random_search.cv_results_["mean_test_score"].max(),
        "std_test_score": random_search.cv_results_["std_test_score"][
            random_search.cv_results_["mean_test_score"].argmax()
        ],
    }

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV score: {cv_scores['mean_test_score']:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_model, best_params, cv_scores, train_time


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
    n_jobs: int = -1,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any], Dict[str, float], float]:
    """
    Train XGBoost CPU Regressor with tree_method="hist" and hyperparameter optimization

    Args:
        X_train: Training features
        y_train: Training target (incident counts)
        cv_splits: List of (train_indices, test_indices) tuples
        n_trials: Number of random search trials
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, cv_scores, train_time_sec)
    """
    logger.info("Training XGBoost CPU (tree_method='hist')...")

    # Hyperparameter distribution
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "objective": ["reg:squarederror", "count:poisson"],  # Try both objectives
    }

    # Create custom CV from splits
    # Use list-of-tuples directly (PredefinedSplit causes leakage in rolling-origin CV)
    cv = [(train_idx, test_idx) for (train_idx, test_idx) in cv_splits]

    # Create model
    model = xgb.XGBRegressor(
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        objective="reg:squarederror",  # Can also try "count:poisson"
    )

    # Use negative RMSE as scorer
    scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )

    # Random search
    start_time = time.time()
    random_search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_trials,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
    )

    random_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    cv_scores = {
        "mean_test_score": random_search.cv_results_["mean_test_score"].max(),
        "std_test_score": random_search.cv_results_["std_test_score"][
            random_search.cv_results_["mean_test_score"].argmax()
        ],
    }

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV score: {cv_scores['mean_test_score']:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_model, best_params, cv_scores, train_time


def train_model_with_hpo(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    hpo_method: str = "grid",
    n_trials: int = 20,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    Generic function to train a model with hyperparameter optimization

    Args:
        model_name: Name of model ('LogisticRegression', 'RandomForest', 'XGBoost')
        X_train: Training features
        y_train: Training target
        cv_splits: List of (train_indices, test_indices) tuples
        hpo_method: 'grid' or 'random'
        n_trials: Number of trials for random search
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, cv_scores, train_time_sec)
    """
    if model_name == "LogisticRegression":
        model, params, scores, train_time = train_logistic_regression(
            X_train, y_train, cv_splits, n_jobs
        )
    elif model_name == "RandomForest":
        model, params, scores, train_time = train_random_forest(
            X_train, y_train, cv_splits, n_trials, n_jobs
        )
    elif model_name == "XGBoost":
        model, params, scores, train_time = train_xgboost(
            X_train, y_train, cv_splits, n_trials, n_jobs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, params, scores, train_time

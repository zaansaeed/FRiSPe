import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



def train_ridge_model(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> make_pipeline:
    """
    Train a Ridge regression model using time series cross-validation.

    Args:
        X (pd.DataFrame): Feature matrix with date index.
        y (pd.Series): Target variable with date index.
        alpha (float): Regularization strength for Ridge regression.

    Returns:
        make_pipeline: Trained Ridge regression model pipeline.
    """
    # Ensure X and y are aligned
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = -np.inf

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create a pipeline with scaling and Ridge regression
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model

def predict_ridge_model(model: make_pipeline, X_new: pd.DataFrame) -> pd.Series:
    """
    Use the trained Ridge regression model to make predictions.

    Args:
        model (make_pipeline): Trained Ridge regression model pipeline.
        X_new (pd.DataFrame): New feature matrix for prediction.

    Returns:
        pd.Series: Predicted values indexed by date.
    """
    predictions = model.predict(X_new)
    return pd.Series(predictions, index=X_new.index)

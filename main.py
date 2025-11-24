from infra.data_pipeline import build_feature_matrix, build_returns_matrix
from forecasting.ridge_model import train_ridge_model, predict_ridge_model
from forecasting.xgb_model import train_xgb_model, predict_xgb_model
import pandas as pd
import numpy as np

if __name__ == "__main__":

    assets = ['AAPL', 'NVDA', 'GOOGL'] 
    build_feature_matrix(assets)
    build_returns_matrix(assets)

    # Load feature and returns matrices
    feature_matrix = pd.read_csv("./data/feature_matrix.csv", index_col=0, parse_dates=True)
    returns_matrix = pd.read_csv("./data/returns_matrix.csv", index_col=0, parse_dates=True)
    target_asset = 'AAPL'
    target_column = f"{target_asset}"    
    X = feature_matrix.loc[:, feature_matrix.columns.str.startswith(target_asset)]
    y = returns_matrix[target_column]
    # Train Ridge model
    ridge_model = train_ridge_model(X, y)
    

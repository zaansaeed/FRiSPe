import pandas as pd
import numpy as np

def compute_risk_signal(df_iv, df_rv, df_corr) -> pd.DataFrame:
    """
    Compute risk signal (VRP z-score) for each asset and return the full
    z-score DataFrame (rows = dates, columns = asset names).

    Returns:
        pd.DataFrame: z-score values indexed by date with one column per asset.
    """
    
    window = 252  # 1 year lookback
    min_periods = 10  # Minimum 10 days
    
    # Ensure same assets in both DataFrames
    common_assets = df_iv.columns.intersection(df_rv.columns)
    # Align indices
    common_index = df_iv.index.intersection(df_rv.index)
    df_iv = df_iv.loc[common_index, common_assets]
    df_rv = df_rv.loc[common_index, common_assets]
    
    
    # Calculate VRP for all assets (vectorized)
    vrp_raw = (df_iv[common_assets] - df_rv[common_assets]) / df_rv[common_assets]
    
    # Drop rows with any NaN in VRP
    #vrp_raw = vrp_raw.dropna()



    # Update common_index to match vrp_raw
    common_index = vrp_raw.index


    # Handle correlation signal
    corr_signal_df = pd.DataFrame(0.0, index=common_index, columns=common_assets)
    
    if df_corr is not None and not df_corr.empty:
        if isinstance(df_corr, pd.DataFrame):
            corr_signal = df_corr.iloc[:, 0]
        else:
            corr_signal = df_corr
        
        # Align correlation with VRP index
        corr_signal = corr_signal.loc[common_index]
        
        # Fill NaNs in correlation
        corr_signal = corr_signal.ffill().bfill().fillna(0)
        
        # Broadcast correlation to all assets
        for asset in common_assets:
            corr_signal_df[asset] = corr_signal.values
    
    # Composite signal for all assets
    composite_signal = (
        1.0 * vrp_raw +
        0 * corr_signal_df # Currently zeroed out; adjust weight as needed
    )



    # Calculate rolling z-scores for all assets (vectorized!)
    rolling_mean = composite_signal.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = composite_signal.rolling(window=window, min_periods=min_periods).std()
    
    zscore_risk = (composite_signal - rolling_mean) / rolling_std

    # Drop any date (row) that has NaN for any asset so returned DataFrame
    # contains only complete dates across all assets.
    zscore_risk = zscore_risk.dropna(how="any")
    
    # Strip timezone and time component from index so dates show as just dates (e.g., 2025-01-23)
    zscore_risk.index = pd.to_datetime(zscore_risk.index).normalize().tz_localize(None).date
    return zscore_risk


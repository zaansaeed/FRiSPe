import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pandas as pd
import os

def get_free_float(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Method 1: Direct float shares
    if 'floatShares' in info:
        return info['floatShares']
    
    # Method 2: Approximate from shares outstanding
    shares_out = info.get('sharesOutstanding', 0)
    held_percent_insiders = info.get('heldPercentInsiders', 0)
    
    # Conservative estimate
    free_float = shares_out * (1 - held_percent_insiders)
    return free_float

def get_spy_weights():
    url = "https://www.ssga.com/us/en/individual/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
    file_name = "spy_daily.xlsx"
    csv_name = "spy_daily.csv"

    response = requests.get(url)
    response.raise_for_status()

    with open(file_name, "wb") as f:
        f.write(response.content)

    df = pd.read_excel(file_name)
    df_filtered = df[["SPDR® S&P 500® ETF Trust", "Unnamed: 4"]]
    df_filtered.columns = ["Ticker", "Weight"]
    df_filtered.to_csv(f"./data/{csv_name}", index=False)

    os.remove(file_name)

def get_etf_flows():
    pass

def compute_flow_signal(df_flows) -> pd.DataFrame:
    """
    df_flows: DataFrame with date index and market names as columns containing flow data.

    Returns:
        pd.DataFrame: z-score values indexed by date with one column per asset.
    """
    SPY_ETF_WEIGHTS_FILE = "./data/spy_daily.csv"
    if not os.path.exists(SPY_ETF_WEIGHTS_FILE):
        get_spy_weights()
    spy_weights = pd.read_csv(SPY_ETF_WEIGHTS_FILE)

    rollingWindow = 252
    flowRatios = pd.DataFrame(index=df_flows.index, columns=df_flows.columns)
    
    for asset in df_flows.columns:
        weight = spy_weights.loc[spy_weights["Ticker"] == asset, "Weight"].values[0] #for now, assume asset is in SPY and use its weight
        assetFreeFloat = get_free_float(asset)
        
        # Vectorized calculation across all dates at once
        assetFlow = pd.Series(0.0, index=df_flows.index)
        for etf, weight in assetETFWeights.items():
            if etf in df_flows.columns:
                assetFlow += df_flows[etf] * weight  
        flowRatios[asset] = assetFlow / assetFreeFloat 
    
    
    # Calculate rolling z-scores for last value only
    rolling_mean = flowRatios.rolling(window=rollingWindow, min_periods=10).mean()
    rolling_std = flowRatios.rolling(window=rollingWindow, min_periods=10).std()
    rolling_std = rolling_std.replace(0, np.nan)
    
    zscore_flow = (flowRatios - rolling_mean) / rolling_std
    
    # Return the full z-score DataFrame (rows = dates, cols = assets).
    # Drop any date rows that contain NaN for any asset so callers receive
    # only complete dates across all assets.
    zscore_flow = zscore_flow.dropna(how="any")

    return zscore_flow


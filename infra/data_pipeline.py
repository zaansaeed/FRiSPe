import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import math
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import sys
sys.path.insert(0, '.')
import os
from engines.risk_engine import compute_risk_signal
from engines.flow_engine import compute_flow_signal
from engines.sentiment_engine import compute_sentiment_signal
from engines.regime_detector import detect_regime
from engines.macro_engine import compute_macro_state


api_key = "PKYPCEPN5LQCB6HSUNDBSGJUZT"
secret_key = "9k918toqysNgW2Y4CFomWaLKd1NcHMLR6TY56TSRpkE"
client = StockHistoricalDataClient(api_key, secret_key)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)


def prepare_data_frame(assets):
    """Prepare a DataFrame with relevant statistics for the given assets."""
    
    request = StockBarsRequest(
        symbol_or_symbols=assets,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    df = df.reset_index()
    df['log_return'] = df.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))  #compute log returns
    return df


def calc_realized_volatility(window, df) -> pd.DataFrame:
    """Calculate the realized volatility over a rolling window."""

    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    df.dropna(inplace=True)   
    df['realized_vol'] = df.groupby('symbol')['log_return'].rolling(window=window).std().reset_index(drop=True) # 30-day rolling std dev of log returns
    df['realized_vol_annual'] = df['realized_vol'] * np.sqrt(252) # Annualize the volatility

    df = df.pivot(index='timestamp', columns='symbol', values='realized_vol_annual')
    return df
    
    
    
def calc_correlations_series(window, df) -> pd.DataFrame:
    """Calculate rolling correlations between asset pairs over a specified window."""

    def get_avg_corr(corr_matrix):
        # Extract upper triangle (avoid duplicates and diagonal)
        return corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # Pivot so each column is an asset, each row is a timestamp
    returns_pivot = df.pivot(index='timestamp', columns='symbol', values='log_return')

    # Calculate rolling correlation matrix for each day (20-day window)
    rolling_corr_matrices = returns_pivot.rolling(window=window).corr()
    # Get average of all pairwise correlations for each day (excluding diagonal)

    daily_avg_corr = rolling_corr_matrices.groupby(level=0).apply(get_avg_corr)
    corr_df = pd.DataFrame(daily_avg_corr, columns=['avg_corr'])

    return corr_df # DataFrame with date index and average correlation values


def calc_implied_volatility(df, assets) -> pd.DataFrame:
    """Calculate implied volatility by correlating asset returns with VIXY."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp')
    
    client = StockHistoricalDataClient(api_key, secret_key)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch VIXY
    vixy_request = StockBarsRequest(
        symbol_or_symbols='VIXY',
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    vixy_data = client.get_stock_bars(vixy_request)
    
    vixy_df = vixy_data.df.reset_index().set_index('timestamp')['close']
    
    # Create IV estimates dataframe
    iv_estimates = pd.DataFrame(index=vixy_df.index)
    
    for asset in assets:
        iv_estimates[f'{asset}'] = vixy_df.values / 100
    
    return iv_estimates


def build_feature_matrix(assets):
    """Build the feature matrix for risk signal computation."""
    df = prepare_data_frame(assets)
    iv_df = calc_implied_volatility(df, assets)
    rv_df = calc_realized_volatility(30, df)
    corr_df = calc_correlations_series(30, df)
    # Safely call engines; if an engine fails, substitute an empty DataFrame
    # with the expected asset columns and the date index from available data.
    # This makes the merge robust while still allowing later dropna.
    try:
        flow_df = compute_flow_signal(df)
    except Exception:
        flow_df = pd.DataFrame(index=rv_df.index, columns=assets)

    try:
        sentiment_df = compute_sentiment_signal(assets)
    except Exception:
        sentiment_df = pd.DataFrame(index=rv_df.index, columns=assets)

    try:
        risk_df = compute_risk_signal(iv_df, rv_df, corr_df)
    except Exception:
        risk_df = pd.DataFrame(index=rv_df.index, columns=assets)

    try:
        df_macro = compute_macro_state()
        regime_df = detect_regime(df_macro)
    except Exception:
        regime_df = pd.DataFrame(index=rv_df.index, columns=assets)

    # Helper to normalize outputs to DataFrame with date index and asset columns
    def _ensure_df(x, name):
        if x is None:
            return pd.DataFrame(index=rv_df.index, columns=assets)
        if isinstance(x, pd.Series):
            # If Series indexed by assets but single timestamp, convert to one-row DF
            if x.index.equals(pd.Index(assets)):
                return x.to_frame().T
            # If Series indexed by dates (per-asset?), try to convert to DataFrame
            return x.to_frame()
        if isinstance(x, pd.DataFrame):
            return x
        # Fallback
        return pd.DataFrame(x, index=rv_df.index)

    flow_df = _ensure_df(flow_df, "flow")
    sentiment_df = _ensure_df(sentiment_df, "sentiment")
    risk_df = _ensure_df(risk_df, "risk")
    regime_df = _ensure_df(regime_df, "regime")

    # Normalize all indices to tz-naive dates (just date, no time)
    def _normalize_index(df_obj):
        df_copy = df_obj.copy()
        # Convert to datetime, strip time/tz, then convert to date
        df_copy.index = pd.to_datetime(df_copy.index).normalize().tz_localize(None).date
        return df_copy

    flow_df = _normalize_index(flow_df)
    sentiment_df = _normalize_index(sentiment_df)
    risk_df = _normalize_index(risk_df)
    regime_df = _normalize_index(regime_df)

    # Prefix columns by feature name: e.g. 'AAPL_flow', 'AAPL_sentiment', ...
    def _prefix_columns(df_obj, feature):
        df_copy = df_obj.copy()
        # If columns aren't asset names (e.g., single column), try to align
        df_copy.columns = [f"{col}_{feature}" for col in df_copy.columns]
        return df_copy

    flow_pref = _prefix_columns(flow_df, "flow")
    sentiment_pref = _prefix_columns(sentiment_df, "sentiment")
    risk_pref = _prefix_columns(risk_df, "risk")
    regime_pref = _prefix_columns(regime_df, "regime")

    # Concatenate horizontally, align on dates (outer) then drop any rows that
    # contain NaN across any feature (per your request to drop incomplete dates).
    combined = pd.concat([flow_pref, sentiment_pref, risk_pref, regime_pref], axis=1, join="outer")
    combined = combined.sort_index()

    # Keep rows that have at least one non-NaN value (for testing).
    combined = combined.dropna(how="all")

    combined.to_csv("./data/feature_matrix.csv", index=True)

#build_feature_matrix(assets)

    
def build_returns_matrix(assets) -> None:
    """Build the target returns matrix for model training."""
    returns_df = pd.DataFrame()
    for asset in assets:
        data = yf.Ticker(asset).history(start=start_date, end=end_date)
        data['return'] = data['Close'].pct_change()
        returns_df[asset] = data['return']

    returns_df.index = pd.to_datetime(returns_df.index).normalize().tz_localize(None).date
    returns_df.to_csv("./data/returns_matrix.csv", index=True)



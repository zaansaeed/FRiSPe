import pandas as pd
import numpy as np

import time
from datetime import datetime, timedelta
import os
from datetime import datetime, timezone

def compute_sentiment_signal(assets, history_file="./data/sentiment_history.csv") -> pd.DataFrame:
    """Calculate rolling z-score sentiment using historical cache.

    This function updates `history_file` with today's average sentiment per
    asset (same behavior as before) and then returns a DataFrame of rolling
    z-scores indexed by date with one column per asset. Any date rows that
    contain NaN for any asset are dropped.
    """
    from massive import RESTClient
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    client = RESTClient("yr8gPIMphs6DxIY2nsPTo29bAVmyr7gz") #massive API to get sentiment data
    sentiment_scores = {"positive": 1, "neutral": 0, "negative": -1}
    sentiment_data = {asset: [] for asset in assets}
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    
    for asset in assets:
        news = []
        try:
            for n in client.list_ticker_news(
                ticker=asset, 
                published_utc_gte=today,
                limit=50
            ):
                news.append(n)
        except Exception as e:
            print(f"Error fetching news for {asset}: {e}")
            time.sleep(10)
            continue
        print(len(news), f"news items fetched for {asset}.")
        for item in news:
            if hasattr(item, 'insights') and item.insights:
                for insight in item.insights:
                    if insight.ticker == asset:
                        score = sentiment_scores.get(insight.sentiment, 0)
                        sentiment_data[asset].append(score)
        
        time.sleep(5)

    # Calculate today's average sentiment per asset
    today_sentiment = {}
    for asset, scores in sentiment_data.items():
        today_sentiment[asset] = np.mean(scores) if scores else 0
    

    # Load or create history; always update today's entry if it exists or append if new
    today_date = datetime.now(timezone.utc).date()
    today_df = pd.DataFrame([today_sentiment], index=[today_date])
    
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file, index_col=0, parse_dates=True)
        # Convert index to dates to handle mixed formats (timestamps vs dates)
        history_df.index = pd.to_datetime(history_df.index).normalize().date
        # Remove duplicates, keeping the last occurrence
        history_df = history_df[~history_df.index.duplicated(keep='last')]
        # Remove today's entry if it exists, then append the new one
        history_df = history_df[history_df.index != today_date]
        history_df = pd.concat([history_df, today_df])
        history_df.to_csv(history_file)
    else:
        history_df = today_df
        history_df.to_csv(history_file)
    
    # Ensure numeric dtype
    history_df = history_df.apply(pd.to_numeric, errors="coerce")

    # Strip timezone and time component from index so dates show as just dates (e.g., 2025-01-23)
    history_df.index = pd.to_datetime(history_df.index).normalize().tz_localize(None).date

    # Calculate rolling z-scores (vectorized) using a 1-year window and
    # minimum periods similar to other engines.
    window = 252
    min_periods = 1
    rolling_mean = history_df.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = history_df.rolling(window=window, min_periods=min_periods).std()

    zscore_df = (history_df - rolling_mean) / rolling_std

    # Drop any date rows that have NaN for any asset so callers receive only
    # complete dates across all assets.
    zscore_df = zscore_df.dropna(how="all")


    return zscore_df


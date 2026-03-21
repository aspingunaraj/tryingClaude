"""
Feature engineering for the ML trade filter.

`add_ml_features` enriches an already-prepared DataFrame (one that already
has vwap, atr, volume_avg, ema9, ema21, adx14, minute_of_day columns) with
the ML-specific features used as model inputs.

`build_training_data` creates (X, y) from a completed backtest run on
training data — no lookahead: features are captured at *entry* time, labels
come from the trade outcome (future data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical feature columns consumed by the ML model
FEATURE_COLS = [
    "vwap_dev",        # VWAP deviation (signed %)
    "vwap_dist_pct",   # absolute VWAP distance (%)
    "vol_ratio",       # volume vs rolling average
    "minute_of_day",   # time-of-day in market candles
    "atr_pct",         # ATR relative to price (intraday volatility proxy)
    "ema_slope",       # (EMA9 - EMA21) / close  — trend direction & strength
    "rolling_std_pct", # 20-period close std / close
    "adx14",           # directional movement strength
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML feature columns to a DataFrame that already contains the base
    indicators (vwap, atr, volume_avg, ema9, ema21, adx14, minute_of_day).

    Returns a copy — does NOT modify df in place.
    """
    df = df.copy()
    close = df["close"].replace(0.0, np.nan)

    df["vwap_dev"]        = (df["close"] - df["vwap"]) / df["vwap"].replace(0.0, np.nan)
    df["vwap_dist_pct"]   = df["vwap_dev"].abs()
    df["vol_ratio"]       = df["volume"] / df["volume_avg"].replace(0.0, np.nan)
    df["atr_pct"]         = df["atr"] / close
    df["ema_slope"]       = (df["ema9"] - df["ema21"]) / close
    df["rolling_std_pct"] = df["close"].rolling(20, min_periods=5).std() / close

    return df


def build_training_data(
    trades_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and binary label series y for ML training.

    Parameters
    ----------
    trades_df   : trades returned by run_backtest (must have entry_time, pnl_pct)
    prepared_df : the same DataFrame used for backtesting, with ML features added

    Returns
    -------
    X : pd.DataFrame  (n_trades × n_features)
    y : pd.Series     (1 = profitable trade, 0 = losing/break-even)

    No lookahead: features are at the candle where the trade was *entered*;
    the label is the trade's final pnl (only available after exit).
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

    feat_lookup = prepared_df.set_index("datetime")
    available_features = [c for c in FEATURE_COLS if c in feat_lookup.columns]

    rows, labels = [], []
    for _, trade in trades_df.iterrows():
        entry_time = trade["entry_time"]
        if entry_time not in feat_lookup.index:
            continue
        feat_row = feat_lookup.loc[entry_time]
        rows.append({col: feat_row[col] for col in available_features})
        labels.append(1 if trade["pnl_pct"] > 0 else 0)

    if not rows:
        return pd.DataFrame(columns=available_features), pd.Series(dtype=int)

    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(labels, dtype=int)
    return X, y

"""Technical indicators: intraday VWAP, ATR, rolling volume average.

All indicators reset at the start of each trading day to avoid cross-day leakage.
"""
import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP = cumsum(typical_price * volume) / cumsum(volume)
    Resets at the start of each calendar day.
    Typical price = (high + low + close) / 3
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]

    vwap = pd.Series(np.nan, index=df.index, dtype=float)

    for _, group in df.groupby("date", sort=False):
        idx      = group.index
        cum_vol  = df.loc[idx, "volume"].cumsum()
        cum_pv   = pv[idx].cumsum()
        vwap[idx] = cum_pv / cum_vol.replace(0, np.nan)

    return vwap


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, with daily reset to avoid overnight gaps inflating ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby("date", sort=False):
        idx = group.index
        atr[idx] = tr[idx].rolling(period, min_periods=1).mean()

    return atr


def compute_rolling_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling average volume across the full series (no daily reset needed)."""
    return df["volume"].rolling(period, min_periods=1).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with VWAP, ATR, rolling volume and minute_of_day added."""
    df = df.copy()
    df["vwap"]         = compute_vwap(df)
    df["atr"]          = compute_atr(df)
    df["volume_avg"]   = compute_rolling_volume(df)
    # 0-indexed candle count within each day — used for time filters
    df["minute_of_day"] = df.groupby("date").cumcount()
    return df

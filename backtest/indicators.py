"""
Technical indicators for the multi-strategy ensemble system.

Intraday (daily reset): VWAP, ATR
Cross-day (full series): VWAP slope, ATR rolling average, RSI, volume average,
                         minute_of_day

VWAP slope is the per-candle rate of change of VWAP over a rolling window —
used for both regime detection (TREND vs RANGE) and ML features.

ATR average (atr_avg) is a rolling mean of ATR across all candles — used for
BREAKOUT regime detection (ATR spike vs recent baseline).

EMA and ADX are removed; the multi-strategy regime uses VWAP slope + ATR spike
instead.
"""
import numpy as np
import pandas as pd

VWAP_SLOPE_WINDOW = 5    # candles used for VWAP slope rolling diff
ATR_AVG_WINDOW    = 20   # candles for rolling ATR baseline (regime detection)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP = cumsum(typical_price × volume) / cumsum(volume).
    Resets at the start of each calendar day.
    Typical price = (high + low + close) / 3.
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
    """
    Average True Range with daily reset to prevent overnight gaps inflating ATR.
    """
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


def compute_vwap_slope(vwap: pd.Series, window: int = VWAP_SLOPE_WINDOW) -> pd.Series:
    """
    Rolling slope of VWAP: (vwap[t] - vwap[t-window]) / window.

    Positive → price centre of gravity rising (uptrend).
    Negative → falling (downtrend).
    Near zero → flat / ranging.
    """
    slope = vwap.diff(window) / window
    return slope.fillna(0.0)


def compute_atr_avg(atr: pd.Series, window: int = ATR_AVG_WINDOW) -> pd.Series:
    """
    Rolling average of ATR over `window` candles (cross-day, no reset).
    Used as a baseline for BREAKOUT regime detection:
      atr > regime_atr_multiplier × atr_avg  →  BREAKOUT
    """
    return atr.rolling(window, min_periods=1).mean()


def compute_rolling_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling average volume (no daily reset)."""
    return df["volume"].rolling(period, min_periods=1).mean()


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI with Wilder's smoothing (alpha = 1/period).
    Kept for use as an ML feature; not used by any strategy signal.
    """
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all indicators added:
      - vwap        (daily reset)
      - atr         (daily reset, period 14)
      - vwap_slope  (rolling diff of VWAP — trend direction)
      - atr_avg     (rolling mean of ATR — volatility baseline)
      - volume_avg  (rolling 20-period volume average)
      - rsi14       (full-series, Wilder smoothing — ML feature only)
      - minute_of_day (0-indexed candle count within each day)
    """
    df = df.copy()
    df["vwap"]         = compute_vwap(df)
    df["atr"]          = compute_atr(df)
    df["vwap_slope"]   = compute_vwap_slope(df["vwap"])
    df["atr_avg"]      = compute_atr_avg(df["atr"])
    df["volume_avg"]   = compute_rolling_volume(df)
    df["rsi14"]        = compute_rsi(df)
    df["minute_of_day"] = df.groupby("date").cumcount()
    return df

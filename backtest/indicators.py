"""
Technical indicators: intraday VWAP, ATR, rolling volume, EMA, ADX.

VWAP and ATR reset daily (intraday indicators).
EMA and ADX are computed on the full series — they require cross-day history
to give meaningful trend/regime signals.
"""
import numpy as np
import pandas as pd


def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average of close price (no daily reset)."""
    return df["close"].ewm(span=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's Average Directional Index (no daily reset).

    Steps:
      1. True Range (TR)
      2. +DM / -DM
      3. Wilder-smooth TR, +DM, -DM  (alpha = 1/period)
      4. +DI = 100 * smooth_plus_DM / smooth_TR
      5. -DI = 100 * smooth_minus_DM / smooth_TR
      6. DX  = 100 * |+DI - -DI| / (+DI + -DI)
      7. ADX = Wilder-smooth DX
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm  = np.where((high - prev_high) > (prev_low - low),
                        np.maximum(high - prev_high, 0.0), 0.0)
    minus_dm = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0.0), 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=df.index, dtype=float)

    # Wilder smoothing: ewm with com = period - 1
    alpha       = 1.0 / period
    smooth_tr   = tr.ewm(alpha=alpha, adjust=False).mean()
    smooth_plus = plus_dm_s.ewm(alpha=alpha,  adjust=False).mean()
    smooth_minus= minus_dm_s.ewm(alpha=alpha, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = 100.0 * smooth_plus  / smooth_tr.replace(0, np.nan)
        minus_di = 100.0 * smooth_minus / smooth_tr.replace(0, np.nan)
        di_sum   = (plus_di + minus_di).replace(0, np.nan)
        dx       = 100.0 * (plus_di - minus_di).abs() / di_sum

    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx


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


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (no daily reset).

    Uses Wilder's smoothing (EWM with alpha=1/period) to match the original RSI
    definition.  Returns values in [0, 100]; NaN for the first few candles.
    """
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all indicators added:
      - VWAP (daily reset), ATR (daily reset), rolling volume avg
      - EMA 9, EMA 21 (full-series)
      - ADX 14, RSI 14 (full-series)
      - minute_of_day
    """
    df = df.copy()
    df["vwap"]         = compute_vwap(df)
    df["atr"]          = compute_atr(df)
    df["volume_avg"]   = compute_rolling_volume(df)
    df["ema9"]         = compute_ema(df, 9)
    df["ema21"]        = compute_ema(df, 21)
    df["adx14"]        = compute_adx(df, 14)
    df["rsi14"]        = compute_rsi(df, 14)
    # 0-indexed candle count within each day — used for time filters
    df["minute_of_day"] = df.groupby("date").cumcount()
    return df

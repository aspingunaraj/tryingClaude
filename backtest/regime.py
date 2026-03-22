"""
Symbol-level regime detection — three-regime classification.

Each candle is labelled as exactly ONE of:

  TREND    — strong directional VWAP slope; activate VWAP Trend Pullback strategy.
  BREAKOUT — ATR spike vs recent baseline; activate Opening Range Breakout strategy.
  RANGE    — default (flat slope, normal ATR); activate VWAP Rejection MR strategy.

Priority order (evaluated per candle):
  1. BREAKOUT: atr > regime_atr_multiplier × atr_avg
  2. TREND:    abs(vwap_slope) > vwap_slope_threshold
  3. RANGE:    everything else

This is NOT market-wide — it is computed from the symbol's own OHLCV data.
The parameters (vwap_slope_threshold, regime_atr_multiplier) come from
StrategyParams and are part of the optimizer search space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_TREND    = "TREND"
REGIME_RANGE    = "RANGE"
REGIME_BREAKOUT = "BREAKOUT"


def detect_regime(
    vwap_slope:          float,
    atr:                 float,
    atr_avg:             float,
    vwap_slope_threshold:  float,
    regime_atr_multiplier: float,
) -> str:
    """
    Classify a single candle into one of three regimes.

    Parameters
    ----------
    vwap_slope            : rolling VWAP slope for this candle
    atr                   : current ATR
    atr_avg               : rolling ATR average (baseline)
    vwap_slope_threshold  : abs(slope) > this → TREND
    regime_atr_multiplier : atr > N × atr_avg → BREAKOUT
    """
    # Priority 1: BREAKOUT — ATR spike
    if atr_avg > 0 and atr > regime_atr_multiplier * atr_avg:
        return REGIME_BREAKOUT

    # Priority 2: TREND — strong directional VWAP slope
    if abs(vwap_slope) > vwap_slope_threshold:
        return REGIME_TREND

    # Default: RANGE
    return REGIME_RANGE


def add_regime(
    df: pd.DataFrame,
    vwap_slope_threshold:  float = 0.0003,
    regime_atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Vectorised regime labelling for an entire DataFrame.

    Requires columns: vwap_slope, atr, atr_avg
    (produced by indicators.add_all_indicators).

    Returns a copy with a 'regime' string column added.
    """
    df = df.copy()

    atr     = df["atr"].fillna(0.0)
    atr_avg = df["atr_avg"].fillna(atr)
    slope   = df["vwap_slope"].fillna(0.0)

    # Vectorised priority logic
    is_breakout = (atr_avg > 0) & (atr > regime_atr_multiplier * atr_avg)
    is_trend    = slope.abs() > vwap_slope_threshold

    regime = np.where(
        is_breakout,
        REGIME_BREAKOUT,
        np.where(is_trend, REGIME_TREND, REGIME_RANGE),
    )
    df["regime"] = regime
    return df


def regime_summary(df: pd.DataFrame) -> dict:
    """Return per-regime candle counts and percentages (informational)."""
    if "regime" not in df.columns:
        return {}
    total = len(df)
    if total == 0:
        return {}
    counts = df["regime"].value_counts()
    return {
        "trend_pct":    round(counts.get(REGIME_TREND,    0) / total * 100, 1),
        "range_pct":    round(counts.get(REGIME_RANGE,    0) / total * 100, 1),
        "breakout_pct": round(counts.get(REGIME_BREAKOUT, 0) / total * 100, 1),
        "n_trend":      int(counts.get(REGIME_TREND,    0)),
        "n_range":      int(counts.get(REGIME_RANGE,    0)),
        "n_breakout":   int(counts.get(REGIME_BREAKOUT, 0)),
    }

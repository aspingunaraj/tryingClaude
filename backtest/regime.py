"""
Market regime detection — rule-based approach.

Labels each candle as either:
  MEAN_REVERTING (0) — ADX is low and EMA trend is flat → VWAP trades favoured
  TRENDING        (1) — strong directional move → avoid VWAP mean reversion

Rules (all conditions must hold for MEAN_REVERTING):
  • ADX14 < adx_threshold          (e.g. 25)   — weak trend
  • |ema_slope| < ema_slope_thresh  (e.g. 3e-4) — flat EMA spread

Everything else is labelled TRENDING.

The regime column is used by run_backtest_ml to skip signals during trending
markets when the regime_filter flag is set in MLConfig.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_MEAN_REVERTING = 0
REGIME_TRENDING       = 1


def add_regime(
    df: pd.DataFrame,
    adx_threshold:       float = 25.0,
    ema_slope_threshold: float = 3e-4,
) -> pd.DataFrame:
    """
    Add a 'regime' column to df.

    Parameters
    ----------
    df                  : DataFrame with adx14 and ema_slope columns
                          (call add_all_indicators + add_ml_features first)
    adx_threshold       : ADX below this → low trend strength
    ema_slope_threshold : |ema_slope| below this → flat EMA → mean-reverting

    Returns a copy with the 'regime' integer column added.
    """
    df = df.copy()

    adx       = df["adx14"].fillna(50.0)          # default to trending when ADX missing
    ema_slope = df["ema_slope"].fillna(0.0).abs()  # already computed by add_ml_features

    df["regime"] = np.where(
        (adx < adx_threshold) & (ema_slope < ema_slope_threshold),
        REGIME_MEAN_REVERTING,
        REGIME_TRENDING,
    ).astype(int)

    return df


def regime_summary(df: pd.DataFrame) -> dict:
    """Return fraction of candles in each regime (informational)."""
    if "regime" not in df.columns:
        return {}
    total = len(df)
    if total == 0:
        return {}
    n_mr = int((df["regime"] == REGIME_MEAN_REVERTING).sum())
    n_tr = int((df["regime"] == REGIME_TRENDING).sum())
    return {
        "mean_reverting_pct": round(n_mr / total * 100, 1),
        "trending_pct":       round(n_tr / total * 100, 1),
        "n_mean_reverting":   n_mr,
        "n_trending":         n_tr,
    }

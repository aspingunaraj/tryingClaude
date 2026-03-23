"""
Candlestick pattern detection on 5-minute OHLCV data.

Each detector function returns a signal string ('long', 'short') or None.
All patterns operate on a slice of the DataFrame passed as individual rows
for clarity and testability.

Trend context (used by Doji, Hammer, Shooting Star):
    uptrend   : close[i] > close[i - TREND_LB]
    downtrend : close[i] < close[i - TREND_LB]

Position size is always 1 share — P&L = exit_price - entry_price (long)
                                         or entry_price - exit_price (short).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

TREND_LB = 5   # candles to look back for local trend context


# ---------------------------------------------------------------------------
# Low-level candle geometry helpers
# ---------------------------------------------------------------------------

def _body(c: pd.Series) -> float:
    return abs(c["close"] - c["open"])

def _upper_wick(c: pd.Series) -> float:
    return c["high"] - max(c["open"], c["close"])

def _lower_wick(c: pd.Series) -> float:
    return min(c["open"], c["close"]) - c["low"]

def _range(c: pd.Series) -> float:
    return c["high"] - c["low"]

def _is_bullish(c: pd.Series) -> bool:
    return c["close"] > c["open"]

def _is_bearish(c: pd.Series) -> bool:
    return c["close"] < c["open"]

def _trend(df: pd.DataFrame, i: int) -> Optional[str]:
    """Simple price-based trend over TREND_LB candles."""
    if i < TREND_LB:
        return None
    delta = df.iloc[i]["close"] - df.iloc[i - TREND_LB]["close"]
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return None


# ---------------------------------------------------------------------------
# Pattern detectors  (return 'long', 'short', or None)
# ---------------------------------------------------------------------------

def detect_engulfing(df: pd.DataFrame, i: int) -> Optional[str]:
    """Bullish or Bearish Engulfing — body of candle i fully contains body of i-1."""
    if i < 1:
        return None
    prev, curr = df.iloc[i - 1], df.iloc[i]
    prev_hi = max(prev["open"], prev["close"])
    prev_lo = min(prev["open"], prev["close"])
    curr_hi = max(curr["open"], curr["close"])
    curr_lo = min(curr["open"], curr["close"])

    if _is_bullish(curr) and _is_bearish(prev) and curr_lo < prev_lo and curr_hi > prev_hi:
        return "long"
    if _is_bearish(curr) and _is_bullish(prev) and curr_hi > prev_hi and curr_lo < prev_lo:
        return "short"
    return None


def detect_hammer(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    Hammer  → long  (after downtrend): small body at top, lower wick ≥ 2× body
    Shooting Star → short (after uptrend): small body at bottom, upper wick ≥ 2× body
    """
    if i < TREND_LB:
        return None
    c = df.iloc[i]
    rng = _range(c)
    if rng == 0:
        return None
    body = _body(c)
    lo_w = _lower_wick(c)
    up_w = _upper_wick(c)

    # Hammer
    if (lo_w >= 2 * body and up_w <= 0.3 * rng and body <= 0.4 * rng
            and _trend(df, i) == "down"):
        return "long"
    # Shooting star
    if (up_w >= 2 * body and lo_w <= 0.3 * rng and body <= 0.4 * rng
            and _trend(df, i) == "up"):
        return "short"
    return None


def detect_doji(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    Doji: body < 10% of range.  Direction based on prior trend.
    After uptrend → short  (potential reversal down)
    After downtrend → long (potential reversal up)
    """
    if i < TREND_LB:
        return None
    c = df.iloc[i]
    rng = _range(c)
    if rng == 0:
        return None
    if _body(c) > 0.1 * rng:
        return None

    t = _trend(df, i)
    if t == "up":
        return "short"
    if t == "down":
        return "long"
    return None


def detect_harami(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    Harami: small candle body inside the prior large candle body.
    Bullish harami (prev bearish, curr bullish) → long
    Bearish harami (prev bullish, curr bearish) → short
    """
    if i < 1:
        return None
    prev, curr = df.iloc[i - 1], df.iloc[i]
    prev_hi = max(prev["open"], prev["close"])
    prev_lo = min(prev["open"], prev["close"])
    curr_hi = max(curr["open"], curr["close"])
    curr_lo = min(curr["open"], curr["close"])
    prev_body = _body(prev)
    curr_body = _body(curr)

    if prev_body == 0:
        return None

    # curr body must be inside prev body AND noticeably smaller
    inside = curr_lo >= prev_lo and curr_hi <= prev_hi
    smaller = curr_body < 0.5 * prev_body

    if inside and smaller:
        if _is_bearish(prev) and _is_bullish(curr):
            return "long"
        if _is_bullish(prev) and _is_bearish(curr):
            return "short"
    return None


def detect_morning_evening_star(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    Morning Star (3-candle) → long
    Evening Star (3-candle) → short
    """
    if i < 2:
        return None
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]

    # Morning star: large bearish, small middle, large bullish closing > midpoint of c1
    if (_is_bearish(c1) and _body(c2) < 0.4 * _body(c1)
            and _is_bullish(c3)
            and c3["close"] > (c1["open"] + c1["close"]) / 2):
        return "long"

    # Evening star: large bullish, small middle, large bearish closing < midpoint of c1
    if (_is_bullish(c1) and _body(c2) < 0.4 * _body(c1)
            and _is_bearish(c3)
            and c3["close"] < (c1["open"] + c1["close"]) / 2):
        return "short"
    return None


def detect_pin_bar(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    Pin Bar: one dominant wick (≥ 60% of total range), small body.
    Long lower wick → long  (rejection of lower prices)
    Long upper wick → short (rejection of higher prices)
    """
    c = df.iloc[i]
    rng = _range(c)
    if rng == 0:
        return None
    lo_w = _lower_wick(c)
    up_w = _upper_wick(c)
    body = _body(c)

    if lo_w >= 0.6 * rng and body <= 0.25 * rng and up_w <= 0.2 * rng:
        return "long"
    if up_w >= 0.6 * rng and body <= 0.25 * rng and lo_w <= 0.2 * rng:
        return "short"
    return None


def detect_marubozu(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    Marubozu: large body, almost no wicks (wicks ≤ 5% of body each).
    Bullish → long, Bearish → short.
    Body must be at least 0.5% of price to filter tiny noise candles.
    """
    c = df.iloc[i]
    body = _body(c)
    if body == 0:
        return None
    rng = _range(c)
    if body < 0.7 * rng:        # wicks account for more than 30% — not a marubozu
        return None
    if body / c["close"] < 0.003:  # body < 0.3% of price — too small
        return None

    if _is_bullish(c):
        return "long"
    if _is_bearish(c):
        return "short"
    return None


# ---------------------------------------------------------------------------
# Master detector
# ---------------------------------------------------------------------------

PATTERN_FUNCS = {
    "engulfing":    detect_engulfing,
    "hammer":       detect_hammer,
    "doji":         detect_doji,
    "harami":       detect_harami,
    "star":         detect_morning_evening_star,
    "pin_bar":      detect_pin_bar,
    "marubozu":     detect_marubozu,
}

PATTERN_LABELS = {
    "engulfing": "Engulfing",
    "hammer":    "Hammer / Shooting Star",
    "doji":      "Doji (context)",
    "harami":    "Harami",
    "star":      "Morning / Evening Star",
    "pin_bar":   "Pin Bar",
    "marubozu":  "Marubozu",
}


def detect_all(df: pd.DataFrame, enabled: list[str]) -> pd.DataFrame:
    """
    Run all enabled pattern detectors row-by-row.

    Returns the input DataFrame with two added columns:
        signal  : 'long' | 'short' | None
        pattern : name of the pattern that fired (or None)
    """
    n = len(df)
    signals  = [None] * n
    patterns = [None] * n

    funcs = [(name, PATTERN_FUNCS[name]) for name in enabled if name in PATTERN_FUNCS]

    for i in range(n):
        for name, fn in funcs:
            sig = fn(df, i)
            if sig is not None:
                signals[i]  = sig
                patterns[i] = name
                break   # first matching pattern wins; avoid double-counting

    out = df.copy()
    out["signal"]  = signals
    out["pattern"] = patterns
    return out

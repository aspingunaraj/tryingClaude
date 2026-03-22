"""5-Min Trend Pullback Engulfing Strategy backtester.

Public API
----------
detect_trend(df, idx, params)       -> 'up' | 'down' | None
detect_pullback(row, trend, params) -> bool
detect_engulfing(prev, curr, trend) -> bool
generate_signal(df, idx, params)    -> dict | None   (rich signal dict)
run_backtest(df, params)            -> {"trades": DataFrame, "equity_curve": Series,
                                        "last_signal": dict | None}

df must already have indicators added by indicators.add_all_indicators:
  vwap, ema, atr, volume_avg, minute_of_day
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .strategy import StrategyParams


# ---------------------------------------------------------------------------
# Signal detection helpers
# ---------------------------------------------------------------------------

def detect_trend(df: pd.DataFrame, idx: int, params: StrategyParams) -> Optional[str]:
    """
    Determine trend at candle `idx`.

    Uptrend:   close > VWAP  AND  close > EMA  AND  close[idx] > close[idx - lookback]
    Downtrend: close < VWAP  AND  close < EMA  AND  close[idx] < close[idx - lookback]
    Otherwise: None
    """
    lookback = params.trend_lookback
    if idx < lookback:
        return None

    row  = df.iloc[idx]
    prev = df.iloc[idx - lookback]

    close = row["close"]
    vwap  = row["vwap"]
    ema   = row["ema"]

    if close > vwap and close > ema and close > prev["close"]:
        return "up"
    if close < vwap and close < ema and close < prev["close"]:
        return "down"
    return None


def detect_pullback(row: pd.Series, trend: str, params: StrategyParams) -> bool:
    """
    Price is within pullback_zone_pct of VWAP or EMA20.

    In an uptrend the price should have pulled back toward (or into) the
    VWAP/EMA zone before the engulfing reversal candle.
    In a downtrend the price should have bounced toward the zone.
    """
    close = row["close"]
    vwap  = row["vwap"]
    ema   = row["ema"]
    zone  = params.pullback_zone_pct

    near_vwap = abs(close - vwap) / vwap < zone if vwap > 0 else False
    near_ema  = abs(close - ema)  / ema  < zone if ema  > 0 else False

    return near_vwap or near_ema


def detect_engulfing(prev: pd.Series, curr: pd.Series, trend: str) -> bool:
    """
    Bullish engulfing (in uptrend) or bearish engulfing (in downtrend).

    Bullish:
      - prev candle bearish  (close < open)
      - curr candle bullish  (close > open)
      - curr body engulfs prev body: curr_open <= prev_close AND curr_close >= prev_open

    Bearish:
      - prev candle bullish  (close > open)
      - curr candle bearish  (close < open)
      - curr body engulfs prev body: curr_open >= prev_close AND curr_close <= prev_open
    """
    po, pc = prev["open"], prev["close"]
    co, cc = curr["open"], curr["close"]

    if trend == "up":
        return pc < po and cc > co and co <= pc and cc >= po

    if trend == "down":
        return pc > po and cc < co and co >= pc and cc <= po

    return False


def _volume_confirmed(df: pd.DataFrame, idx: int, params: StrategyParams) -> bool:
    """Current candle volume > mean of the preceding `volume_lookback` candles."""
    lookback = params.volume_lookback
    if idx < lookback:
        return False
    curr_vol = df.iloc[idx]["volume"]
    avg_vol  = df.iloc[idx - lookback: idx]["volume"].mean()
    return avg_vol > 0 and curr_vol > avg_vol


def generate_signal(
    df: pd.DataFrame,
    idx: int,
    params: StrategyParams,
) -> Optional[dict]:
    """
    Return a rich signal dict if all conditions are met at candle `idx`, else None.

    Checks (all must pass):
      1. Session window (minute_of_day in [session_start, session_end])
      2. ATR / close > atr_sideways_pct  (not a flat market)
      3. Trend  (up or down)
      4. Pullback: previous candle was in the VWAP/EMA zone
      5. Engulfing candle at `idx`
      6. Volume confirmation at `idx`
    """
    min_look = max(params.trend_lookback, params.volume_lookback) + 1
    if idx < min_look:
        return None

    row  = df.iloc[idx]
    prev = df.iloc[idx - 1]

    # 1. Session filter
    mod = int(row["minute_of_day"])
    trade_window_active = params.session_start_candle <= mod <= params.session_end_candle
    if not trade_window_active:
        return None

    # 2. ATR sideways filter
    close = float(row["close"])
    atr   = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
    if close > 0 and atr / close < params.atr_sideways_pct:
        return None

    # 3. Trend
    trend = detect_trend(df, idx, params)
    if trend is None:
        return None

    # 4. Pullback on previous candle
    pullback_valid = detect_pullback(prev, trend, params)
    if not pullback_valid:
        return None

    # 5. Engulfing on current candle
    engulfing_detected = detect_engulfing(prev, row, trend)
    if not engulfing_detected:
        return None

    # 6. Volume
    volume_condition = _volume_confirmed(df, idx, params)
    if not volume_condition:
        return None

    signal = "long" if trend == "up" else "short"

    # Compute entry, SL, TP
    slippage = params.slippage
    if signal == "long":
        entry_price = close * (1 + slippage)
        stop_loss   = float(row["low"]) * (1 - slippage)
    else:
        entry_price = close * (1 - slippage)
        stop_loss   = float(row["high"]) * (1 + slippage)

    risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return None

    target_price = (
        entry_price + params.risk_reward_ratio * risk
        if signal == "long"
        else entry_price - params.risk_reward_ratio * risk
    )

    return {
        "timestamp":           str(row["datetime"]),
        "signal":              signal,
        "trend":               trend,
        "entry_price":         round(entry_price,  4),
        "stop_loss":           round(stop_loss,    4),
        "target_price":        round(target_price, 4),
        "risk_reward":         round(params.risk_reward_ratio, 2),
        "volume_condition":    bool(volume_condition),
        "pullback_valid":      bool(pullback_valid),
        "engulfing_detected":  bool(engulfing_detected),
        "vwap_value":          round(float(row["vwap"]), 4),
        "ema_20":              round(float(row["ema"]),  4),
        "trade_window_active": bool(trade_window_active),
        "atr":                 round(atr, 4),
        "minute_of_day":       mod,
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, params: StrategyParams) -> dict:
    """
    Run the 5-Min Trend Pullback Engulfing Strategy.

    Parameters
    ----------
    df     : 5-min DataFrame with indicators already added
    params : StrategyParams

    Returns
    -------
    {
      "trades":       pd.DataFrame,
      "equity_curve": pd.Series,
      "last_signal":  dict | None,   # most recent valid signal in df
    }
    """
    trades: list    = []
    equity: float   = 0.0
    equity_pts: list = [0.0]

    in_trade: bool   = False
    trade_info: dict = {}
    last_signal: Optional[dict] = None

    cost = params.round_trip_cost()

    # Pre-compute per-day candle counts for EOD detection
    day_counts = df.groupby("date").size().to_dict()

    n = len(df)

    for idx in range(1, n):
        row = df.iloc[idx]

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            direction   = trade_info["direction"]
            entry_price = trade_info["entry_price"]
            stop_loss   = trade_info["stop_loss"]
            take_profit = trade_info["take_profit"]
            trailing_sl = trade_info["trailing_sl"]
            start_idx   = trade_info["start_idx"]

            candles_held = idx - start_idx
            day_size     = day_counts.get(row["date"], 9999)
            mod          = int(row["minute_of_day"])

            # Update trailing stop to current EMA
            if params.use_trailing_stop:
                ema_now = float(row["ema"])
                if direction == "long":
                    trailing_sl = max(trailing_sl, ema_now)
                else:
                    trailing_sl = min(trailing_sl, ema_now)
                trade_info["trailing_sl"] = trailing_sl
                effective_sl = trailing_sl
            else:
                effective_sl = stop_loss

            low  = float(row["low"])
            high = float(row["high"])

            exit_price:  Optional[float] = None
            exit_reason: Optional[str]   = None

            if direction == "long":
                if low <= effective_sl:
                    exit_price  = effective_sl
                    exit_reason = "stop"
                elif high >= take_profit:
                    exit_price  = take_profit
                    exit_reason = "target"
            else:
                if high >= effective_sl:
                    exit_price  = effective_sl
                    exit_reason = "stop"
                elif low <= take_profit:
                    exit_price  = take_profit
                    exit_reason = "target"

            # Force-exit: max holding or EOD
            if exit_price is None:
                if candles_held >= params.max_holding_candles:
                    exit_price  = float(row["close"])
                    exit_reason = "max_hold"
                elif mod >= day_size - params.eod_buffer_candles:
                    exit_price  = float(row["close"])
                    exit_reason = "eod"

            if exit_price is not None:
                if direction == "long":
                    pnl_pct = (exit_price - entry_price) / entry_price - cost
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price - cost

                equity += pnl_pct
                equity_pts.append(equity)

                trades.append({
                    "entry_time":      trade_info["entry_time"],
                    "exit_time":       row["datetime"],
                    "direction":       direction,
                    "entry_price":     round(entry_price,  4),
                    "exit_price":      round(exit_price,   4),
                    "stop_loss":       round(stop_loss,    4),
                    "take_profit":     round(take_profit,  4),
                    "pnl_pct":         round(pnl_pct,      6),
                    "exit_reason":     exit_reason,
                    "candles_held":    candles_held,
                    "holding_minutes": candles_held * 5,
                })

                in_trade   = False
                trade_info = {}
                continue

        # ── Look for new signal ───────────────────────────────────────────────
        if not in_trade:
            sig = generate_signal(df, idx, params)
            if sig is not None:
                last_signal = sig

                ep = sig["entry_price"]
                sl = sig["stop_loss"]
                tp = sig["target_price"]
                direction = sig["signal"]

                risk = abs(ep - sl)
                if risk > 0:
                    in_trade   = True
                    trade_info = {
                        "direction":   direction,
                        "entry_price": ep,
                        "stop_loss":   sl,
                        "take_profit": tp,
                        "trailing_sl": sl,
                        "entry_time":  row["datetime"],
                        "start_idx":   idx,
                    }

        equity_pts.append(equity)

    trades_df = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "stop_loss", "take_profit", "pnl_pct", "exit_reason",
            "candles_held", "holding_minutes",
        ])
    )
    equity_curve = pd.Series(equity_pts, dtype=float)

    return {
        "trades":       trades_df,
        "equity_curve": equity_curve,
        "last_signal":  last_signal,
    }

"""
Intraday VWAP Mean-Reversion backtesting engine.

Design choices:
 - Loop-based (not fully vectorized) to handle per-candle stop/take-profit
   checks correctly without lookahead bias.  At ~45 k candles per symbol the
   loop runs in < 1 s on any modern CPU.
 - One position at a time (configurable in future via allow_multiple flag).
 - EOD force-exit prevents overnight holding.
 - Slippage + commission applied symmetrically on both legs.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .strategy import StrategyParams


# ---------------------------------------------------------------------------
# Internal trade record
# ---------------------------------------------------------------------------

class _Trade:
    __slots__ = (
        "entry_time", "exit_time",
        "direction",                 # +1 long / -1 short
        "entry_price", "exit_price",
        "exit_reason",
        "pnl_pct", "holding_minutes",
        "_entry_minute", "_entry_date",
    )

    def __init__(self, entry_time, direction: int, entry_price: float,
                 entry_minute: int, entry_date):
        self.entry_time     = entry_time
        self.direction      = direction
        self.entry_price    = entry_price
        self._entry_minute  = entry_minute
        self._entry_date    = entry_date
        self.exit_time      = None
        self.exit_price     = None
        self.exit_reason    = None
        self.pnl_pct        = None
        self.holding_minutes = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, params: StrategyParams) -> Dict:
    """
    Run a single backtest pass on `df` (must already contain VWAP, ATR,
    volume_avg, minute_of_day columns — call indicators.add_all_indicators first).

    Returns
    -------
    {
      "trades":       pd.DataFrame  – one row per closed trade,
      "equity_curve": pd.Series     – cumulative fractional PnL, indexed by datetime,
      "n_trades":     int,
      "final_equity": float,
    }
    """
    cost = params.slippage + params.commission  # one-way; applied on both legs

    # Build per-day candle count for the time_close_filter
    day_candle_count: Dict = df.groupby("date").size().to_dict()

    records  = df.to_dict("records")
    trades: List[_Trade] = []
    position: Optional[_Trade] = None
    equity   = 0.0
    eq_values: List[float] = []

    for row in records:
        dt      = row["datetime"]
        close   = row["close"]
        vwap    = row["vwap"]
        volume  = row["volume"]
        vol_avg = row["volume_avg"]
        minute  = row["minute_of_day"]
        date    = row["date"]
        n_min   = day_candle_count.get(date, 375)

        if pd.isna(vwap) or vwap == 0:
            eq_values.append(equity)
            continue

        # ── Manage open position ──────────────────────────────────────────
        if position is not None:

            # Safety: force close if somehow a position carries overnight
            if position._entry_date != date:
                _close_position(position, close, dt, "overnight_close", cost)
                equity += position.pnl_pct
                trades.append(position)
                position = None
                eq_values.append(equity)
                continue

            elapsed    = minute - position._entry_minute
            exit_price: Optional[float] = None
            reason:     Optional[str]   = None

            if position.direction == 1:   # Long
                if   close >= vwap:
                    exit_price, reason = close, "vwap"
                elif close <= position.entry_price * (1 - params.stop_loss):
                    exit_price, reason = close, "stop_loss"
                elif close >= position.entry_price * (1 + params.take_profit):
                    exit_price, reason = close, "take_profit"
            else:                          # Short
                if   close <= vwap:
                    exit_price, reason = close, "vwap"
                elif close >= position.entry_price * (1 + params.stop_loss):
                    exit_price, reason = close, "stop_loss"
                elif close <= position.entry_price * (1 - params.take_profit):
                    exit_price, reason = close, "take_profit"

            # Max holding timeout
            if exit_price is None and elapsed >= params.max_holding:
                exit_price, reason = close, "timeout"

            # End-of-day forced exit (last time_close_filter candles)
            if exit_price is None and minute >= n_min - params.time_close_filter - 1:
                exit_price, reason = close, "eod"

            if exit_price is not None:
                _close_position(position, exit_price, dt, reason, cost)
                equity  += position.pnl_pct
                trades.append(position)
                position = None

        # ── Check for new entry (only when flat) ─────────────────────────
        if position is None:
            # Time filters
            if minute < params.time_open_filter:
                eq_values.append(equity)
                continue
            if minute >= n_min - params.time_close_filter:
                eq_values.append(equity)
                continue
            # Volume filter
            if vol_avg > 0 and volume < vol_avg * params.volume_filter:
                eq_values.append(equity)
                continue

            # VWAP deviation signals
            dev = (close - vwap) / vwap
            if dev < -params.threshold:          # price is below VWAP → go long
                position = _Trade(dt, +1, close, minute, date)
            elif dev > params.threshold:          # price is above VWAP → go short
                position = _Trade(dt, -1, close, minute, date)

        eq_values.append(equity)

    # ── Force-close any remaining position at the last bar ───────────────
    if position is not None and records:
        last  = records[-1]
        _close_position(position, last["close"], last["datetime"], "eod_final", cost)
        equity += position.pnl_pct
        trades.append(position)

    trades_df    = _trades_to_df(trades)
    equity_curve = pd.Series(eq_values, index=df["datetime"], dtype=float)

    return {
        "trades":       trades_df,
        "equity_curve": equity_curve,
        "n_trades":     len(trades),
        "final_equity": equity,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_position(pos: _Trade, exit_price: float, exit_time,
                    reason: str, cost: float) -> None:
    """Compute PnL with slippage/commission and stamp the trade."""
    if pos.direction == 1:
        eff_entry = pos.entry_price * (1 + cost)
        eff_exit  = exit_price      * (1 - cost)
    else:
        eff_entry = pos.entry_price * (1 - cost)
        eff_exit  = exit_price      * (1 + cost)

    pnl_pct = pos.direction * (eff_exit - eff_entry) / eff_entry

    pos.exit_price      = exit_price
    pos.exit_time       = exit_time
    pos.exit_reason     = reason
    pos.pnl_pct         = pnl_pct
    pos.holding_minutes = getattr(pos, "_entry_minute", 0)   # recalc below


def _trades_to_df(trades: List[_Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "exit_reason",
            "pnl_pct", "holding_minutes",
        ])
    rows = []
    for t in trades:
        rows.append({
            "entry_time":      t.entry_time,
            "exit_time":       t.exit_time,
            "direction":       "long" if t.direction == 1 else "short",
            "entry_price":     t.entry_price,
            "exit_price":      t.exit_price,
            "exit_reason":     t.exit_reason,
            "pnl_pct":         t.pnl_pct,
            "holding_minutes": t.holding_minutes,
        })
    return pd.DataFrame(rows)

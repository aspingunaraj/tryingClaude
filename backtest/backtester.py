"""
Intraday VWAP Mean-Reversion backtesting engine.

Design choices:
 - Loop-based (not fully vectorized) to handle per-candle stop/take-profit
   checks correctly without lookahead bias.  At ~45 k candles per symbol the
   loop runs in < 1 s on any modern CPU.
 - One position at a time (configurable in future via allow_multiple flag).
 - EOD force-exit prevents overnight holding.
 - Slippage + commission applied symmetrically on both legs.

Entry lenses implemented (14-lens framework):
  Lens 1 VWAP Signal Quality  — threshold, vwap_dev_max, vwap_sustained_candles
  Lens 2 Momentum (RSI)       — rsi_oversold / rsi_overbought / rsi_reversal_required
  Lens 3 Volume               — volume_filter, volume_declining_on_move
  Lens 4 Time / Session       — time_open/close_filter, preferred_session_start/end
  Lens 5 Market Regime        — adx_trend_cap
  Lens 6 Position Management  — max_daily_entries, cooldown_after_loss_minutes

Exit lenses implemented:
  Lens 1 Static Price         — stop_loss, take_profit
  Lens 2 Dynamic Price        — trailing_stop_pct, breakeven_trigger_pct
  Lens 3 Time                 — max_holding, session_hard_exit_buffer
  Lens 4 Regime Change        — volatility_exit_multiplier
  Lens 5 Signal Invalidation  — signal_flip_exit_pct
  Lens 6 Momentum Deterioration — exit_rsi_cross
  Lens 7 Partial Scaling      — partial_exit_ratio, partial_exit_trigger_pct
"""
from __future__ import annotations

import math
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
        "direction",                  # +1 long / -1 short
        "entry_price", "exit_price",
        "exit_reason",
        "pnl_pct", "holding_minutes",
        "_entry_minute", "_entry_date",
        "ml_prob", "position_size",   # ML fields (default 0.5 / 1.0 for base runs)
        "_pnl_raw",                   # raw per-unit pnl before size scaling
        # ── new lens tracking ────────────────────────────────────────────────
        "entry_atr",                  # ATR at entry (Lens 4 Exit: volatility exit)
        "peak_pnl",                   # best raw PnL seen (Lens 2 Exit: trailing / breakeven)
        "breakeven_set",              # flag: stop has been moved to breakeven
        "partial_exited",             # flag: partial exit has already fired
        "partial_pnl_booked",         # PnL already booked from partial close
        "partial_ratio_done",         # fraction of position already closed (0.0 if none)
    )

    def __init__(self, entry_time, direction: int, entry_price: float,
                 entry_minute: int, entry_date,
                 ml_prob: float = 0.5, position_size: float = 1.0):
        self.entry_time          = entry_time
        self.direction           = direction
        self.entry_price         = entry_price
        self._entry_minute       = entry_minute
        self._entry_date         = entry_date
        self.ml_prob             = ml_prob
        self.position_size       = position_size
        self.exit_time           = None
        self.exit_price          = None
        self.exit_reason         = None
        self.pnl_pct             = None
        self.holding_minutes     = None
        self._pnl_raw            = None
        # new fields
        self.entry_atr           = float("nan")
        self.peak_pnl            = 0.0
        self.breakeven_set       = False
        self.partial_exited      = False
        self.partial_pnl_booked  = 0.0
        self.partial_ratio_done  = 0.0


# ---------------------------------------------------------------------------
# Public API — base backtest
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, params: StrategyParams) -> Dict:
    """
    Run a single backtest pass on `df` (must already contain VWAP, ATR,
    volume_avg, rsi14, adx14, minute_of_day columns — call
    indicators.add_all_indicators first).

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

    day_candle_count: Dict = df.groupby("date").size().to_dict()

    records  = df.to_dict("records")
    trades: List[_Trade] = []
    position: Optional[_Trade] = None
    equity   = 0.0
    eq_values: List[float] = []

    # ── per-loop tracking state ───────────────────────────────────────────────
    consecutive_below    = 0    # Lens 1: candles consecutively below VWAP
    consecutive_above    = 0    # Lens 1: candles consecutively above VWAP
    daily_entries: Dict  = {}   # Lens 6: {date_str: entry_count}
    last_loss_exit_dt    = None # Lens 6: datetime of last losing trade exit
    _prev_rsi            = float("nan")  # for RSI reversal + cross checks
    _prev_volume         = float("nan")  # for volume declining check

    for row in records:
        dt      = row["datetime"]
        close   = row["close"]
        vwap    = row["vwap"]
        volume  = row["volume"]
        vol_avg = row["volume_avg"]
        minute  = row["minute_of_day"]
        date    = row["date"]
        n_min   = day_candle_count.get(date, 375)
        rsi_now = row.get("rsi14", float("nan"))
        atr_now = row.get("atr",   float("nan"))

        # Capture prev values before updating (used in checks below)
        prev_rsi    = _prev_rsi
        prev_volume = _prev_volume

        # Update rolling state for NEXT iteration (before any continue)
        _prev_rsi    = rsi_now
        _prev_volume = volume

        if pd.isna(vwap) or vwap == 0:
            eq_values.append(equity)
            continue

        # ── Update VWAP sustained-candle counters ─────────────────────────
        dev_signed = (close - vwap) / vwap
        if dev_signed < 0:
            consecutive_below += 1
            consecutive_above  = 0
        elif dev_signed > 0:
            consecutive_above += 1
            consecutive_below  = 0
        else:
            consecutive_below = 0
            consecutive_above = 0

        # ── Manage open position ──────────────────────────────────────────
        if position is not None:

            # Safety: force close if position somehow carries overnight
            if position._entry_date != date:
                _close_position(position, close, dt, "overnight_close", cost)
                equity += position.pnl_pct
                if position.pnl_pct < 0:
                    last_loss_exit_dt = dt
                trades.append(position)
                position = None
                eq_values.append(equity)
                continue

            elapsed = minute - position._entry_minute

            # Raw per-unit PnL at current price (no cost) for dynamic checks
            if position.direction == 1:
                raw_pnl = (close - position.entry_price) / position.entry_price
            else:
                raw_pnl = (position.entry_price - close) / position.entry_price
            position.peak_pnl = max(position.peak_pnl, raw_pnl)

            exit_price: Optional[float] = None
            reason:     Optional[str]   = None

            # Lens 1 Exit — Static Price (VWAP reversion, stop, take-profit)
            if position.direction == 1:
                if   close >= vwap:
                    exit_price, reason = close, "vwap"
                elif close <= position.entry_price * (1 - params.stop_loss):
                    exit_price, reason = close, "stop_loss"
                elif close >= position.entry_price * (1 + params.take_profit):
                    exit_price, reason = close, "take_profit"
            else:
                if   close <= vwap:
                    exit_price, reason = close, "vwap"
                elif close >= position.entry_price * (1 + params.stop_loss):
                    exit_price, reason = close, "stop_loss"
                elif close <= position.entry_price * (1 - params.take_profit):
                    exit_price, reason = close, "take_profit"

            # Lens 2 Exit — Trailing stop (trail from peak PnL)
            if exit_price is None and params.trailing_stop_pct > 0:
                if position.peak_pnl > 0 and raw_pnl < position.peak_pnl - params.trailing_stop_pct:
                    exit_price, reason = close, "trailing_stop"

            # Lens 2 Exit — Breakeven stop (move stop to entry once threshold hit)
            if exit_price is None and params.breakeven_trigger_pct > 0:
                if not position.breakeven_set and position.peak_pnl >= params.breakeven_trigger_pct:
                    position.breakeven_set = True
                if position.breakeven_set and raw_pnl < 0:
                    exit_price, reason = close, "breakeven_stop"

            # Lens 3 Exit — Session hard exit buffer (force close before EOD zone)
            if exit_price is None and params.session_hard_exit_buffer > 0:
                hard_exit_at = n_min - params.time_close_filter - params.session_hard_exit_buffer - 1
                if minute >= hard_exit_at:
                    exit_price, reason = close, "session_hard_exit"

            # Lens 4 Exit — Volatility regime change
            if exit_price is None and params.volatility_exit_multiplier > 0:
                if (not math.isnan(atr_now) and not math.isnan(position.entry_atr)
                        and position.entry_atr > 0
                        and atr_now > params.volatility_exit_multiplier * position.entry_atr):
                    exit_price, reason = close, "volatility_exit"

            # Lens 5 Exit — Signal invalidation (thesis failing: deviation extending)
            if exit_price is None and params.signal_flip_exit_pct > 0:
                flip_threshold = params.threshold + params.signal_flip_exit_pct
                if position.direction == 1 and dev_signed < -flip_threshold:
                    exit_price, reason = close, "signal_flip"
                elif position.direction == -1 and dev_signed > flip_threshold:
                    exit_price, reason = close, "signal_flip"

            # Lens 6 Exit — RSI crosses 50 against the trade (momentum shift)
            if exit_price is None and params.exit_rsi_cross:
                if not math.isnan(rsi_now) and not math.isnan(prev_rsi):
                    if position.direction == 1 and rsi_now < 50 and prev_rsi >= 50:
                        exit_price, reason = close, "rsi_cross"
                    elif position.direction == -1 and rsi_now > 50 and prev_rsi <= 50:
                        exit_price, reason = close, "rsi_cross"

            # Lens 7 Exit — Partial scaling (book part of position at intermediate target)
            if exit_price is None and params.partial_exit_ratio > 0 and not position.partial_exited:
                partial_trigger = params.take_profit * params.partial_exit_trigger_pct
                if raw_pnl >= partial_trigger:
                    position.partial_exited     = True
                    position.partial_ratio_done = params.partial_exit_ratio
                    # Compute net partial PnL with costs applied to the partial exit
                    if position.direction == 1:
                        eff_entry = position.entry_price * (1 + cost)
                        eff_pex   = close               * (1 - cost)
                        ppu       = (eff_pex - eff_entry) / eff_entry
                    else:
                        eff_entry = position.entry_price * (1 - cost)
                        eff_pex   = close               * (1 + cost)
                        ppu       = (eff_entry - eff_pex) / eff_entry
                    position.partial_pnl_booked = ppu * params.partial_exit_ratio
                    equity += position.partial_pnl_booked

            # Lens 3 Exit — Max holding timeout
            if exit_price is None and elapsed >= params.max_holding:
                exit_price, reason = close, "timeout"

            # Lens 3 Exit — EOD forced exit
            if exit_price is None and minute >= n_min - params.time_close_filter - 1:
                exit_price, reason = close, "eod"

            if exit_price is not None:
                _close_position(position, exit_price, dt, reason, cost)
                remaining = 1.0 - position.partial_ratio_done
                equity   += position.pnl_pct * remaining
                if (position.partial_pnl_booked + position.pnl_pct * remaining) < 0:
                    last_loss_exit_dt = dt
                trades.append(position)
                position = None

        # ── Check for new entry (only when flat) ─────────────────────────────
        if position is None:

            # Lens 4 Entry — Time filters (existing)
            if minute < params.time_open_filter:
                eq_values.append(equity)
                continue
            if minute >= n_min - params.time_close_filter:
                eq_values.append(equity)
                continue

            # Lens 4 Entry — Preferred session window
            if params.preferred_session_start > 0 and minute < params.preferred_session_start:
                eq_values.append(equity)
                continue
            if params.preferred_session_end > 0 and minute > params.preferred_session_end:
                eq_values.append(equity)
                continue

            # Lens 3 Entry — Volume filter (existing)
            if vol_avg > 0 and volume < vol_avg * params.volume_filter:
                eq_values.append(equity)
                continue

            # Lens 3 Entry — Volume declining on move
            if params.volume_declining_on_move and not math.isnan(prev_volume):
                if volume >= prev_volume:
                    eq_values.append(equity)
                    continue

            # Lens 1 Entry — VWAP deviation signal (core)
            dev = dev_signed
            long_signal  = dev < -params.threshold
            short_signal = dev >  params.threshold
            if not long_signal and not short_signal:
                eq_values.append(equity)
                continue

            # Lens 1 Entry — VWAP deviation max (price not too extended)
            if params.vwap_dev_max > 0 and abs(dev) > params.vwap_dev_max:
                eq_values.append(equity)
                continue

            # Lens 1 Entry — Sustained candles away from VWAP
            if params.vwap_sustained_candles > 1:
                if long_signal  and consecutive_below < params.vwap_sustained_candles:
                    eq_values.append(equity)
                    continue
                if short_signal and consecutive_above < params.vwap_sustained_candles:
                    eq_values.append(equity)
                    continue

            # Lens 2 Entry — RSI confirmation
            if not math.isnan(rsi_now):
                if long_signal  and rsi_now >= params.rsi_oversold:
                    eq_values.append(equity)
                    continue
                if short_signal and rsi_now <= params.rsi_overbought:
                    eq_values.append(equity)
                    continue
                # RSI must already be turning in the trade direction
                if params.rsi_reversal_required and not math.isnan(prev_rsi):
                    if long_signal  and rsi_now <= prev_rsi:   # must be rising
                        eq_values.append(equity)
                        continue
                    if short_signal and rsi_now >= prev_rsi:   # must be falling
                        eq_values.append(equity)
                        continue

            # Lens 5 Entry — Market regime (ADX trend cap)
            if params.adx_trend_cap > 0:
                adx_val = row.get("adx14", float("nan"))
                if not math.isnan(adx_val) and adx_val > params.adx_trend_cap:
                    eq_values.append(equity)
                    continue

            # Lens 6 Entry — Max daily entries
            day_key = str(date)
            if params.max_daily_entries > 0:
                if daily_entries.get(day_key, 0) >= params.max_daily_entries:
                    eq_values.append(equity)
                    continue

            # Lens 6 Entry — Cooldown after a losing trade
            if params.cooldown_after_loss_minutes > 0 and last_loss_exit_dt is not None:
                elapsed_mins = (dt - last_loss_exit_dt).total_seconds() / 60
                if elapsed_mins < params.cooldown_after_loss_minutes:
                    eq_values.append(equity)
                    continue

            # ── Open position ─────────────────────────────────────────────
            direction = +1 if long_signal else -1
            position  = _Trade(dt, direction, close, minute, date)
            position.entry_atr = atr_now
            daily_entries[day_key] = daily_entries.get(day_key, 0) + 1

        eq_values.append(equity)

    # ── Force-close any remaining position at the last bar ───────────────────
    if position is not None and records:
        last = records[-1]
        _close_position(position, last["close"], last["datetime"], "eod_final", cost)
        remaining = 1.0 - position.partial_ratio_done
        equity   += position.pnl_pct * remaining
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
# ML-enhanced backtest
# ---------------------------------------------------------------------------

def run_backtest_ml(
    df,
    params,
    model=None,
    ml_config=None,
    sizer=None,
) -> Dict:
    """
    ML-enhanced backtest.  Extends `run_backtest` with three ML gates applied
    at each potential entry signal:

      1. Regime filter    — skip signals in TRENDING market (if enabled)
      2. Trade filter     — skip if model probability < threshold
      3. Position sizing  — scale equity contribution by confidence

    All 14 entry/exit lenses from `run_backtest` are also applied here.
    """
    from .feature_engineering import FEATURE_COLS
    from .regime import REGIME_TRENDING
    from .position_sizing import MLConfig, PositionSizer

    if ml_config is None:
        ml_config = MLConfig(enabled=False)
    if sizer is None:
        sizer = PositionSizer()

    cost = params.slippage + params.commission

    day_candle_count: Dict = df.groupby("date").size().to_dict()
    records  = df.to_dict("records")
    trades: List[_Trade] = []
    position: Optional[_Trade] = None
    equity   = 0.0
    eq_values: List[float] = []

    avail_features = [c for c in FEATURE_COLS if c in df.columns]

    # ── per-loop tracking state ───────────────────────────────────────────────
    consecutive_below    = 0
    consecutive_above    = 0
    daily_entries: Dict  = {}
    last_loss_exit_dt    = None
    _prev_rsi            = float("nan")
    _prev_volume         = float("nan")

    for row in records:
        dt      = row["datetime"]
        close   = row["close"]
        vwap    = row["vwap"]
        volume  = row["volume"]
        vol_avg = row["volume_avg"]
        minute  = row["minute_of_day"]
        date    = row["date"]
        n_min   = day_candle_count.get(date, 375)
        rsi_now = row.get("rsi14", float("nan"))
        atr_now = row.get("atr",   float("nan"))

        prev_rsi    = _prev_rsi
        prev_volume = _prev_volume
        _prev_rsi    = rsi_now
        _prev_volume = volume

        if pd.isna(vwap) or vwap == 0:
            eq_values.append(equity)
            continue

        dev_signed = (close - vwap) / vwap
        if dev_signed < 0:
            consecutive_below += 1
            consecutive_above  = 0
        elif dev_signed > 0:
            consecutive_above += 1
            consecutive_below  = 0
        else:
            consecutive_below = 0
            consecutive_above = 0

        # ── Manage open position ──────────────────────────────────────────────
        if position is not None:
            if position._entry_date != date:
                _close_position(position, close, dt, "overnight_close", cost)
                equity += position.pnl_pct * position.position_size
                if position.pnl_pct < 0:
                    last_loss_exit_dt = dt
                trades.append(position)
                position = None
                eq_values.append(equity)
                continue

            elapsed = minute - position._entry_minute

            if position.direction == 1:
                raw_pnl = (close - position.entry_price) / position.entry_price
            else:
                raw_pnl = (position.entry_price - close) / position.entry_price
            position.peak_pnl = max(position.peak_pnl, raw_pnl)

            exit_price_: Optional[float] = None
            reason_:    Optional[str]   = None

            # Lens 1 Exit — Static Price
            if position.direction == 1:
                if   close >= vwap:
                    exit_price_, reason_ = close, "vwap"
                elif close <= position.entry_price * (1 - params.stop_loss):
                    exit_price_, reason_ = close, "stop_loss"
                elif close >= position.entry_price * (1 + params.take_profit):
                    exit_price_, reason_ = close, "take_profit"
            else:
                if   close <= vwap:
                    exit_price_, reason_ = close, "vwap"
                elif close >= position.entry_price * (1 + params.stop_loss):
                    exit_price_, reason_ = close, "stop_loss"
                elif close <= position.entry_price * (1 - params.take_profit):
                    exit_price_, reason_ = close, "take_profit"

            # Lens 2 Exit — Trailing stop
            if exit_price_ is None and params.trailing_stop_pct > 0:
                if position.peak_pnl > 0 and raw_pnl < position.peak_pnl - params.trailing_stop_pct:
                    exit_price_, reason_ = close, "trailing_stop"

            # Lens 2 Exit — Breakeven stop
            if exit_price_ is None and params.breakeven_trigger_pct > 0:
                if not position.breakeven_set and position.peak_pnl >= params.breakeven_trigger_pct:
                    position.breakeven_set = True
                if position.breakeven_set and raw_pnl < 0:
                    exit_price_, reason_ = close, "breakeven_stop"

            # Lens 3 Exit — Session hard exit buffer
            if exit_price_ is None and params.session_hard_exit_buffer > 0:
                hard_exit_at = n_min - params.time_close_filter - params.session_hard_exit_buffer - 1
                if minute >= hard_exit_at:
                    exit_price_, reason_ = close, "session_hard_exit"

            # Lens 4 Exit — Volatility regime change
            if exit_price_ is None and params.volatility_exit_multiplier > 0:
                if (not math.isnan(atr_now) and not math.isnan(position.entry_atr)
                        and position.entry_atr > 0
                        and atr_now > params.volatility_exit_multiplier * position.entry_atr):
                    exit_price_, reason_ = close, "volatility_exit"

            # Lens 5 Exit — Signal invalidation
            if exit_price_ is None and params.signal_flip_exit_pct > 0:
                flip_threshold = params.threshold + params.signal_flip_exit_pct
                if position.direction == 1 and dev_signed < -flip_threshold:
                    exit_price_, reason_ = close, "signal_flip"
                elif position.direction == -1 and dev_signed > flip_threshold:
                    exit_price_, reason_ = close, "signal_flip"

            # Lens 6 Exit — RSI momentum deterioration
            if exit_price_ is None and params.exit_rsi_cross:
                if not math.isnan(rsi_now) and not math.isnan(prev_rsi):
                    if position.direction == 1 and rsi_now < 50 and prev_rsi >= 50:
                        exit_price_, reason_ = close, "rsi_cross"
                    elif position.direction == -1 and rsi_now > 50 and prev_rsi <= 50:
                        exit_price_, reason_ = close, "rsi_cross"

            # Lens 7 Exit — Partial scaling
            if exit_price_ is None and params.partial_exit_ratio > 0 and not position.partial_exited:
                partial_trigger = params.take_profit * params.partial_exit_trigger_pct
                if raw_pnl >= partial_trigger:
                    position.partial_exited     = True
                    position.partial_ratio_done = params.partial_exit_ratio
                    if position.direction == 1:
                        eff_entry = position.entry_price * (1 + cost)
                        eff_pex   = close               * (1 - cost)
                        ppu       = (eff_pex - eff_entry) / eff_entry
                    else:
                        eff_entry = position.entry_price * (1 - cost)
                        eff_pex   = close               * (1 + cost)
                        ppu       = (eff_entry - eff_pex) / eff_entry
                    position.partial_pnl_booked = ppu * params.partial_exit_ratio * position.position_size
                    equity += position.partial_pnl_booked
                    # Reduce remaining position size proportionally
                    position.position_size *= (1.0 - params.partial_exit_ratio)

            # Lens 3 Exit — Max holding timeout
            if exit_price_ is None and elapsed >= params.max_holding:
                exit_price_, reason_ = close, "timeout"

            # Lens 3 Exit — EOD forced exit
            if exit_price_ is None and minute >= n_min - params.time_close_filter - 1:
                exit_price_, reason_ = close, "eod"

            if exit_price_ is not None:
                _close_position(position, exit_price_, dt, reason_, cost)
                equity += position.pnl_pct * position.position_size
                blended = position.partial_pnl_booked + position.pnl_pct * position.position_size
                if blended < 0:
                    last_loss_exit_dt = dt
                trades.append(position)
                position = None

        # ── Check for new entry ───────────────────────────────────────────────
        if position is None:

            # Lens 4 Entry — Time filters
            if minute < params.time_open_filter:
                eq_values.append(equity)
                continue
            if minute >= n_min - params.time_close_filter:
                eq_values.append(equity)
                continue

            # Lens 4 Entry — Preferred session window
            if params.preferred_session_start > 0 and minute < params.preferred_session_start:
                eq_values.append(equity)
                continue
            if params.preferred_session_end > 0 and minute > params.preferred_session_end:
                eq_values.append(equity)
                continue

            # Lens 3 Entry — Volume filter
            if vol_avg > 0 and volume < vol_avg * params.volume_filter:
                eq_values.append(equity)
                continue

            # Lens 3 Entry — Volume declining on move
            if params.volume_declining_on_move and not math.isnan(prev_volume):
                if volume >= prev_volume:
                    eq_values.append(equity)
                    continue

            # Lens 1 Entry — VWAP deviation signal
            dev = dev_signed
            long_signal  = dev < -params.threshold
            short_signal = dev >  params.threshold
            if not long_signal and not short_signal:
                eq_values.append(equity)
                continue

            # Lens 1 Entry — VWAP deviation max
            if params.vwap_dev_max > 0 and abs(dev) > params.vwap_dev_max:
                eq_values.append(equity)
                continue

            # Lens 1 Entry — Sustained candles
            if params.vwap_sustained_candles > 1:
                if long_signal  and consecutive_below < params.vwap_sustained_candles:
                    eq_values.append(equity)
                    continue
                if short_signal and consecutive_above < params.vwap_sustained_candles:
                    eq_values.append(equity)
                    continue

            # Lens 2 Entry — RSI confirmation
            if not math.isnan(rsi_now):
                if long_signal  and rsi_now >= params.rsi_oversold:
                    eq_values.append(equity)
                    continue
                if short_signal and rsi_now <= params.rsi_overbought:
                    eq_values.append(equity)
                    continue
                if params.rsi_reversal_required and not math.isnan(prev_rsi):
                    if long_signal  and rsi_now <= prev_rsi:
                        eq_values.append(equity)
                        continue
                    if short_signal and rsi_now >= prev_rsi:
                        eq_values.append(equity)
                        continue

            # Lens 5 Entry — ADX trend cap
            if params.adx_trend_cap > 0:
                adx_val = row.get("adx14", float("nan"))
                if not math.isnan(adx_val) and adx_val > params.adx_trend_cap:
                    eq_values.append(equity)
                    continue

            # Lens 6 Entry — Max daily entries
            day_key = str(date)
            if params.max_daily_entries > 0:
                if daily_entries.get(day_key, 0) >= params.max_daily_entries:
                    eq_values.append(equity)
                    continue

            # Lens 6 Entry — Cooldown after loss
            if params.cooldown_after_loss_minutes > 0 and last_loss_exit_dt is not None:
                elapsed_mins = (dt - last_loss_exit_dt).total_seconds() / 60
                if elapsed_mins < params.cooldown_after_loss_minutes:
                    eq_values.append(equity)
                    continue

            direction = +1 if long_signal else -1

            # ── ML gate 1: regime filter ──────────────────────────────────────
            if ml_config.enabled and ml_config.regime_filter:
                if row.get("regime", REGIME_TRENDING) == REGIME_TRENDING:
                    eq_values.append(equity)
                    continue

            # ── ML gate 2 + 3: probability filter & sizing ────────────────────
            ml_prob       = 0.5
            position_size = 1.0

            if ml_config.enabled and model is not None:
                feat_row  = pd.DataFrame([{c: row.get(c, float("nan"))
                                           for c in avail_features}])
                ml_prob   = float(model.predict_proba(feat_row)[0])

                if ml_prob < ml_config.filter_threshold:
                    eq_values.append(equity)
                    continue

                position_size = sizer.get_size(ml_prob)
                if position_size == 0.0:
                    eq_values.append(equity)
                    continue

            position = _Trade(dt, direction, close, minute, date,
                              ml_prob=ml_prob, position_size=position_size)
            position.entry_atr = atr_now
            daily_entries[day_key] = daily_entries.get(day_key, 0) + 1

        eq_values.append(equity)

    # Force-close remaining position
    if position is not None and records:
        last = records[-1]
        _close_position(position, last["close"], last["datetime"], "eod_final", cost)
        equity += position.pnl_pct * position.position_size
        trades.append(position)

    # Stamp raw pnl before size-adjusting the main column
    for t in trades:
        t._pnl_raw = t.pnl_pct
        t.pnl_pct  = t.pnl_pct * t.position_size

    trades_df    = _trades_to_df_ml(trades)
    equity_curve = pd.Series(eq_values, index=df["datetime"], dtype=float)

    return {
        "trades":       trades_df,
        "equity_curve": equity_curve,
        "n_trades":     len(trades),
        "final_equity": equity,
    }


def _trades_to_df_ml(trades: List[_Trade]) -> pd.DataFrame:
    """trades_to_df variant that also writes pnl_raw_pct."""
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "exit_reason",
            "pnl_pct", "pnl_raw_pct", "holding_minutes",
            "ml_prob", "position_size",
        ])
    rows = []
    for t in trades:
        # pnl_pct at this point = per_unit * remaining_position_size (size-adjusted)
        # partial_pnl_booked was already added to equity but needs to appear in record
        blended_pnl = t.partial_pnl_booked + t.pnl_pct
        rows.append({
            "entry_time":      t.entry_time,
            "exit_time":       t.exit_time,
            "direction":       "long" if t.direction == 1 else "short",
            "entry_price":     t.entry_price,
            "exit_price":      t.exit_price,
            "exit_reason":     t.exit_reason,
            "pnl_pct":         blended_pnl,
            "pnl_raw_pct":     getattr(t, "_pnl_raw", t.pnl_pct),
            "holding_minutes": t.holding_minutes,
            "ml_prob":         t.ml_prob,
            "position_size":   t.position_size,
        })
    return pd.DataFrame(rows)


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
    pos.holding_minutes = getattr(pos, "_entry_minute", 0)


def _trades_to_df(trades: List[_Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "exit_reason",
            "pnl_pct", "holding_minutes", "ml_prob", "position_size",
        ])
    rows = []
    for t in trades:
        # Blend partial pnl (already booked) with remaining-portion final pnl
        remaining    = 1.0 - t.partial_ratio_done
        blended_pnl  = t.partial_pnl_booked + t.pnl_pct * remaining
        rows.append({
            "entry_time":      t.entry_time,
            "exit_time":       t.exit_time,
            "direction":       "long" if t.direction == 1 else "short",
            "entry_price":     t.entry_price,
            "exit_price":      t.exit_price,
            "exit_reason":     t.exit_reason,
            "pnl_pct":         blended_pnl,
            "holding_minutes": t.holding_minutes,
            "ml_prob":         t.ml_prob,
            "position_size":   t.position_size,
        })
    return pd.DataFrame(rows)

"""
Multi-Strategy Intraday Ensemble Backtester.

Three strategies, each active only in its designated regime:

  Strategy 1 — VWAP Trend Pullback   (TREND regime)
    Long : uptrend + price above VWAP + pulling back within pullback_distance
           + current close > previous candle high (bullish confirmation)
    Short: downtrend + price below VWAP + pulling back within pullback_distance
           + current close < previous candle low  (bearish confirmation)

  Strategy 2 — Opening Range Breakout (BREAKOUT regime)
    Long : close breaks above opening-range high + breakout_buffer
    Short: close breaks below opening-range low  - breakout_buffer
    One trade per direction per day; ORB must be set (opening range complete).

  Strategy 3 — VWAP Rejection Mean Reversion (RANGE regime)
    Long : price below VWAP; previous candle bearish + current candle bullish
           (rejection of further downside) + close > prev_close
    Short: price above VWAP; previous candle bullish + current candle bearish
           (failure to hold above VWAP) + close < prev_close

Exit logic (common to all strategies):
  1. ATR-based stop-loss  : entry_price ± atr_stop_multiplier × ATR
  2. Risk-reward TP       : stop distance × risk_reward_ratio
  3. Max holding time     : max_holding_minutes candles
  4. EOD force-exit       : eod_buffer_candles before end of day

Design:
  - Loop-based (no vectorisation) → correct per-candle stop/TP without lookahead.
  - One position at a time per symbol.
  - Slippage + commission applied symmetrically on both entry and exit legs.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .strategy import StrategyParams
from .regime   import detect_regime, REGIME_TREND, REGIME_RANGE, REGIME_BREAKOUT


# ---------------------------------------------------------------------------
# Internal trade record
# ---------------------------------------------------------------------------

class _Trade:
    __slots__ = (
        "entry_time", "exit_time",
        "direction",
        "entry_price", "exit_price",
        "stop_price",  "tp_price",
        "exit_reason",
        "pnl_pct", "holding_minutes",
        "_entry_minute", "_entry_date",
        "strategy", "regime",
        "ml_prob", "position_size",
    )

    def __init__(
        self,
        entry_time,
        direction:     int,
        entry_price:   float,
        stop_price:    float,
        tp_price:      float,
        entry_minute:  int,
        entry_date,
        strategy:      str,
        regime:        str,
        ml_prob:       float = 0.5,
        position_size: float = 1.0,
    ):
        self.entry_time    = entry_time
        self.direction     = direction
        self.entry_price   = entry_price
        self.stop_price    = stop_price
        self.tp_price      = tp_price
        self._entry_minute = entry_minute
        self._entry_date   = entry_date
        self.strategy      = strategy
        self.regime        = regime
        self.ml_prob       = ml_prob
        self.position_size = position_size

        self.exit_time     = None
        self.exit_price    = None
        self.exit_reason   = None
        self.pnl_pct       = None
        self.holding_minutes = None


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _adjusted_entry(price: float, direction: int, cost: float) -> float:
    """Entry price after slippage + commission: long pays more, short receives less."""
    return price * (1.0 + direction * cost)


def _adjusted_exit(price: float, direction: int, cost: float) -> float:
    """Exit price after slippage + commission: long receives less, short pays more."""
    return price * (1.0 - direction * cost)


def _compute_pnl(trade: _Trade, exit_price: float, cost: float) -> float:
    adj_entry = _adjusted_entry(trade.entry_price, trade.direction, cost)
    adj_exit  = _adjusted_exit(exit_price,        trade.direction, cost)
    return trade.direction * (adj_exit - adj_entry) / adj_entry


# ---------------------------------------------------------------------------
# Strategy signal functions
# ---------------------------------------------------------------------------

def _signal_trend_pullback(
    row:      dict,
    prev_row: dict,
    params:   StrategyParams,
) -> Optional[int]:
    """
    VWAP Trend Pullback signal.
    Returns +1 (long), -1 (short), or None.
    """
    close = row["close"]
    vwap  = row["vwap"]
    slope = row["vwap_slope"]

    if not vwap or vwap == 0:
        return None

    dev = (close - vwap) / vwap   # signed fractional deviation

    # LONG: uptrend confirmed by positive slope, price above VWAP but within
    # pullback distance, bullish close (breaks above previous candle high).
    if (
        slope > params.vwap_slope_threshold
        and 0 < dev < params.pullback_distance
        and close > prev_row["high"]
    ):
        return +1

    # SHORT: mirror conditions — downtrend, price below VWAP, bearish close.
    if (
        slope < -params.vwap_slope_threshold
        and -params.pullback_distance < dev < 0
        and close < prev_row["low"]
    ):
        return -1

    return None


def _signal_orb(
    row:            dict,
    orb_high:       float,
    orb_low:        float,
    params:         StrategyParams,
    long_done:      bool,
    short_done:     bool,
) -> Optional[int]:
    """
    Opening Range Breakout signal.
    Returns +1, -1, or None.  Each direction fires at most once per day.
    """
    close = row["close"]

    if not long_done and close > orb_high * (1.0 + params.breakout_buffer):
        return +1
    if not short_done and close < orb_low * (1.0 - params.breakout_buffer):
        return -1

    return None


def _signal_mean_reversion(
    row:      dict,
    prev_row: dict,
    params:   StrategyParams,
) -> Optional[int]:
    """
    VWAP Rejection Mean Reversion signal.

    Long  (below VWAP, downside rejected):
      prev candle bearish AND current candle bullish AND close > prev_close
      → rejection of further downside; expect bounce toward VWAP.

    Short (above VWAP, upside rejected):
      prev candle bullish AND current candle bearish AND close < prev_close
      → failure to hold above VWAP; expect reversion toward VWAP.

    Returns +1, -1, or None.
    """
    close  = row["close"]
    open_  = row["open"]
    vwap   = row["vwap"]

    if not vwap or vwap == 0:
        return None

    prev_close = prev_row["close"]
    prev_open  = prev_row["open"]

    # LONG: price below VWAP, prior candle was bearish (selling),
    # current candle reversed bullish (rejection of downside).
    if (
        close < vwap
        and prev_close < prev_open   # previous candle: bearish
        and close > open_            # current candle:  bullish
        and close > prev_close       # upward momentum
    ):
        return +1

    # SHORT: price above VWAP, prior candle bullish (buying attempt),
    # current candle reversed bearish (failure to hold above VWAP).
    if (
        close > vwap
        and prev_close > prev_open   # previous candle: bullish
        and close < open_            # current candle:  bearish
        and close < prev_close       # downward momentum
    ):
        return -1

    return None


# ---------------------------------------------------------------------------
# Exit check
# ---------------------------------------------------------------------------

def _check_exit(
    row:          dict,
    trade:        _Trade,
    current_minute: int,
    params:       StrategyParams,
) -> Optional[str]:
    """
    Check all exit conditions for an open position.
    Returns the exit reason string or None (stay in trade).
    """
    close = row["close"]

    # 1. Stop-loss
    if trade.direction == +1 and close <= trade.stop_price:
        return "stop_loss"
    if trade.direction == -1 and close >= trade.stop_price:
        return "stop_loss"

    # 2. Take-profit
    if trade.direction == +1 and close >= trade.tp_price:
        return "take_profit"
    if trade.direction == -1 and close <= trade.tp_price:
        return "take_profit"

    # 3. Max holding time
    holding = current_minute - trade._entry_minute
    if holding >= params.max_holding_minutes:
        return "max_holding"

    return None


# ---------------------------------------------------------------------------
# Trade-to-dict helper
# ---------------------------------------------------------------------------

def _trade_to_dict(trade: _Trade, exit_price: float, exit_reason: str,
                   holding_minutes: int, pnl_pct: float) -> dict:
    return {
        "entry_time":     trade.entry_time,
        "exit_time":      trade.exit_time,
        "direction":      trade.direction,
        "entry_price":    trade.entry_price,
        "exit_price":     exit_price,
        "stop_price":     trade.stop_price,
        "tp_price":       trade.tp_price,
        "exit_reason":    exit_reason,
        "pnl_pct":        pnl_pct,
        "holding_minutes": holding_minutes,
        "strategy":       trade.strategy,
        "regime":         trade.regime,
        "ml_prob":        trade.ml_prob,
        "position_size":  trade.position_size,
    }


# ---------------------------------------------------------------------------
# Shared per-candle processing logic
# ---------------------------------------------------------------------------

_TRADE_COLS = [
    "entry_time", "exit_time", "direction", "entry_price", "exit_price",
    "stop_price", "tp_price", "exit_reason", "pnl_pct", "holding_minutes",
    "strategy", "regime", "ml_prob", "position_size",
]


def _make_empty_result() -> Dict:
    return {
        "trades":       pd.DataFrame(columns=_TRADE_COLS),
        "equity_curve": pd.Series(dtype=float),
        "n_trades":     0,
        "final_equity": 0.0,
    }


# ---------------------------------------------------------------------------
# Public API — base backtest
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, params: StrategyParams) -> Dict:
    """
    Run a full candle-by-candle backtest with the three-strategy ensemble.

    `df` must already contain indicators from indicators.add_all_indicators:
      vwap, atr, atr_avg, vwap_slope, volume_avg, rsi14, minute_of_day

    Returns
    -------
    {
      "trades":       pd.DataFrame  – one row per closed trade,
      "equity_curve": pd.Series     – cumulative fractional PnL,
      "n_trades":     int,
      "final_equity": float,
    }
    """
    if df is None or len(df) == 0:
        return _make_empty_result()

    cost    = params.cost_per_side()
    records = df.to_dict("records")
    n       = len(records)

    # State
    trades:   List[dict]        = []
    position: Optional[_Trade]  = None
    equity    = 0.0
    eq_values: List[float]      = []

    # Daily state
    current_day   = None
    orb_high      = 0.0
    orb_low       = float("inf")
    orb_set       = False
    orb_long_done = False
    orb_short_done = False
    day_total_candles = 0  # total candles in current day (computed on first candle)

    # Precompute per-day candle count for EOD detection
    day_counts: Dict = df.groupby("date").size().to_dict()

    for i in range(n):
        row    = records[i]
        dt     = row.get("datetime", i)
        day    = row["date"]
        minute = row.get("minute_of_day", i)
        close  = row["close"]
        vwap   = row.get("vwap", 0)

        # ── Daily reset ──────────────────────────────────────────────────────
        if day != current_day:
            # Safety: force-close any position left open (shouldn't happen if
            # EOD logic fires correctly, but guards against edge cases).
            if position is not None:
                pnl = _compute_pnl(position, close, cost)
                equity += pnl * position.position_size
                position.exit_time = dt
                trades.append(
                    _trade_to_dict(position, close, "eod_force",
                                   minute - position._entry_minute, pnl)
                )
                position = None

            current_day    = day
            orb_high       = row["high"]
            orb_low        = row["low"]
            orb_set        = False
            orb_long_done  = False
            orb_short_done = False
            day_total_candles = day_counts.get(day, 375)

        # ── Update opening range (before it is set) ──────────────────────────
        if not orb_set:
            orb_high = max(orb_high, row["high"])
            orb_low  = min(orb_low,  row["low"])
            if minute >= params.opening_range_minutes - 1:
                orb_set = True

        # Skip candles with invalid VWAP
        if not vwap or vwap == 0 or np.isnan(vwap):
            eq_values.append(equity)
            continue

        # ── EOD detection ────────────────────────────────────────────────────
        candles_left_in_day = day_total_candles - minute - 1
        is_eod = candles_left_in_day < params.eod_buffer_candles

        # ── Exit logic ───────────────────────────────────────────────────────
        if position is not None:
            exit_reason = _check_exit(row, position, minute, params)

            if exit_reason is None and is_eod:
                exit_reason = "eod"

            if exit_reason:
                pnl = _compute_pnl(position, close, cost)
                equity += pnl * position.position_size
                position.exit_time = dt
                trades.append(
                    _trade_to_dict(
                        position, close, exit_reason,
                        minute - position._entry_minute, pnl,
                    )
                )
                position = None

        eq_values.append(equity)

        # ── Entry logic (skip if in position or in EOD buffer) ───────────────
        if position is not None or is_eod:
            continue

        # Regime detection for this candle
        regime = detect_regime(
            vwap_slope           = row.get("vwap_slope", 0.0),
            atr                  = row.get("atr", 0.0),
            atr_avg              = row.get("atr_avg", 0.0),
            vwap_slope_threshold = params.vwap_slope_threshold,
            regime_atr_multiplier= params.regime_atr_multiplier,
        )

        # Previous candle (same day only; skip on first candle of day)
        prev_row = records[i - 1] if (i > 0 and records[i - 1]["date"] == day) else None

        signal   = None
        strategy = ""

        if regime == REGIME_TREND and prev_row is not None:
            signal   = _signal_trend_pullback(row, prev_row, params)
            strategy = "TREND_PULLBACK"

        elif regime == REGIME_BREAKOUT and orb_set:
            signal   = _signal_orb(row, orb_high, orb_low, params,
                                   orb_long_done, orb_short_done)
            strategy = "ORB"

        elif regime == REGIME_RANGE and prev_row is not None:
            signal   = _signal_mean_reversion(row, prev_row, params)
            strategy = "MEAN_REVERSION"

        if signal is not None:
            atr       = row.get("atr", close * 0.005) or close * 0.005
            stop_dist = params.atr_stop_multiplier * atr
            tp_dist   = stop_dist * params.risk_reward_ratio

            position = _Trade(
                entry_time   = dt,
                direction    = signal,
                entry_price  = close,
                stop_price   = close - signal * stop_dist,
                tp_price     = close + signal * tp_dist,
                entry_minute = minute,
                entry_date   = day,
                strategy     = strategy,
                regime       = regime,
            )

            # Track ORB direction usage
            if strategy == "ORB":
                if signal == +1:
                    orb_long_done  = True
                else:
                    orb_short_done = True

    # Force-close any position remaining at the very end of data
    if position is not None and records:
        last  = records[-1]
        close = last["close"]
        pnl   = _compute_pnl(position, close, cost)
        equity += pnl * position.position_size
        trades.append(
            _trade_to_dict(
                position, close, "final_eod",
                last.get("minute_of_day", 0) - position._entry_minute,
                pnl,
            )
        )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=_TRADE_COLS)

    return {
        "trades":       trades_df,
        "equity_curve": pd.Series(eq_values, dtype=float),
        "n_trades":     len(trades),
        "final_equity": equity,
    }


# ---------------------------------------------------------------------------
# Public API — ML-enhanced backtest
# ---------------------------------------------------------------------------

def run_backtest_ml(
    df,
    params,
    model,
    ml_config,
    sizer=None,
) -> Dict:
    """
    ML-enhanced backtest.

    Identical to run_backtest but adds two optional gates at entry:
      Gate 1: Regime label filter (skip BREAKOUT or TREND if configured)
      Gate 2: ML probability filter  (skip if model confidence < threshold)
      Gate 3: Position sizing by ML confidence (via PositionSizer)

    All exit logic is identical to the base backtest.
    """
    from .feature_engineering import add_ml_features, FEATURE_COLS

    if df is None or len(df) == 0:
        return _make_empty_result()

    # Enrich with ML features; keep column names stable
    df_feat = add_ml_features(df)
    records = df_feat.to_dict("records")
    n       = len(records)

    cost    = params.cost_per_side()
    trades: List[dict]       = []
    position: Optional[_Trade] = None
    equity  = 0.0
    eq_values: List[float]   = []

    current_day   = None
    orb_high      = 0.0
    orb_low       = float("inf")
    orb_set       = False
    orb_long_done = False
    orb_short_done = False

    day_counts: Dict = df_feat.groupby("date").size().to_dict()

    for i in range(n):
        row    = records[i]
        dt     = row.get("datetime", i)
        day    = row["date"]
        minute = row.get("minute_of_day", i)
        close  = row["close"]
        vwap   = row.get("vwap", 0)

        if day != current_day:
            if position is not None:
                pnl = _compute_pnl(position, close, cost)
                equity += pnl * position.position_size
                position.exit_time = dt
                trades.append(
                    _trade_to_dict(position, close, "eod_force",
                                   minute - position._entry_minute, pnl)
                )
                position = None

            current_day    = day
            orb_high       = row["high"]
            orb_low        = row["low"]
            orb_set        = False
            orb_long_done  = False
            orb_short_done = False

        if not orb_set:
            orb_high = max(orb_high, row["high"])
            orb_low  = min(orb_low,  row["low"])
            if minute >= params.opening_range_minutes - 1:
                orb_set = True

        if not vwap or vwap == 0 or np.isnan(vwap):
            eq_values.append(equity)
            continue

        day_total = day_counts.get(day, 375)
        candles_left = day_total - minute - 1
        is_eod = candles_left < params.eod_buffer_candles

        if position is not None:
            exit_reason = _check_exit(row, position, minute, params)
            if exit_reason is None and is_eod:
                exit_reason = "eod"

            if exit_reason:
                pnl = _compute_pnl(position, close, cost)
                equity += pnl * position.position_size
                position.exit_time = dt
                trades.append(
                    _trade_to_dict(
                        position, close, exit_reason,
                        minute - position._entry_minute, pnl,
                    )
                )
                position = None

        eq_values.append(equity)

        if position is not None or is_eod:
            continue

        regime = detect_regime(
            vwap_slope           = row.get("vwap_slope", 0.0),
            atr                  = row.get("atr", 0.0),
            atr_avg              = row.get("atr_avg", 0.0),
            vwap_slope_threshold = params.vwap_slope_threshold,
            regime_atr_multiplier= params.regime_atr_multiplier,
        )

        prev_row = records[i - 1] if (i > 0 and records[i - 1]["date"] == day) else None

        signal   = None
        strategy = ""

        if regime == REGIME_TREND and prev_row is not None:
            signal   = _signal_trend_pullback(row, prev_row, params)
            strategy = "TREND_PULLBACK"
        elif regime == REGIME_BREAKOUT and orb_set:
            signal   = _signal_orb(row, orb_high, orb_low, params,
                                   orb_long_done, orb_short_done)
            strategy = "ORB"
        elif regime == REGIME_RANGE and prev_row is not None:
            signal   = _signal_mean_reversion(row, prev_row, params)
            strategy = "MEAN_REVERSION"

        if signal is None:
            continue

        # ── ML gates ────────────────────────────────────────────────────────
        ml_prob       = 0.5
        position_size = 1.0

        if model is not None and getattr(model, "_trained", False):
            try:
                feat_row = pd.DataFrame([{c: row.get(c, 0.0) for c in FEATURE_COLS}])
                probs    = model.predict_proba(feat_row)
                ml_prob  = float(probs[0])
            except Exception:
                ml_prob = 0.5

            if ml_prob < ml_config.filter_threshold:
                # ML says skip this trade
                if strategy == "ORB":
                    if signal == +1:
                        orb_long_done  = True
                    else:
                        orb_short_done = True
                continue

            if sizer is not None:
                position_size = sizer.get_size(ml_prob)

        # ── Open position ────────────────────────────────────────────────────
        atr       = row.get("atr", close * 0.005) or close * 0.005
        stop_dist = params.atr_stop_multiplier * atr
        tp_dist   = stop_dist * params.risk_reward_ratio

        position = _Trade(
            entry_time    = dt,
            direction     = signal,
            entry_price   = close,
            stop_price    = close - signal * stop_dist,
            tp_price      = close + signal * tp_dist,
            entry_minute  = minute,
            entry_date    = day,
            strategy      = strategy,
            regime        = regime,
            ml_prob       = ml_prob,
            position_size = position_size,
        )

        if strategy == "ORB":
            if signal == +1:
                orb_long_done  = True
            else:
                orb_short_done = True

    if position is not None and records:
        last  = records[-1]
        close = last["close"]
        pnl   = _compute_pnl(position, close, cost)
        equity += pnl * position.position_size
        trades.append(
            _trade_to_dict(
                position, close, "final_eod",
                last.get("minute_of_day", 0) - position._entry_minute,
                pnl,
            )
        )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=_TRADE_COLS)

    return {
        "trades":       trades_df,
        "equity_curve": pd.Series(eq_values, dtype=float),
        "n_trades":     len(trades),
        "final_equity": equity,
    }

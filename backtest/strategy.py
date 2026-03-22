"""Strategy parameters and parameter-space definitions.

StrategyParams is a plain dataclass — the backtester and optimizer both consume it.
PARAM_BOUNDS defines the search space for optimisation (core params only).

The 14-lens parameter set is organised by phase (Entry / Exit) and lens:

  ENTRY
  ├── Lens 1: VWAP Signal Quality      threshold, vwap_dev_max, vwap_sustained_candles
  ├── Lens 2: Momentum (RSI)           rsi_oversold, rsi_overbought, rsi_reversal_required
  ├── Lens 3: Volume Confirmation      volume_filter, volume_declining_on_move
  ├── Lens 4: Time / Session           time_open_filter, time_close_filter,
  │                                    preferred_session_start, preferred_session_end
  ├── Lens 5: Market Regime            adx_trend_cap
  └── Lens 6: Position Management      max_daily_entries, cooldown_after_loss_minutes

  EXIT
  ├── Lens 1: Static Price             stop_loss, take_profit
  ├── Lens 2: Dynamic Price            trailing_stop_pct, breakeven_trigger_pct
  ├── Lens 3: Time                     max_holding, session_hard_exit_buffer
  ├── Lens 4: Regime Change            volatility_exit_multiplier
  ├── Lens 5: Signal Invalidation      signal_flip_exit_pct
  ├── Lens 6: Momentum Deterioration   exit_rsi_cross
  └── Lens 7: Partial Scaling          partial_exit_ratio, partial_exit_trigger_pct
"""
from dataclasses import dataclass, fields as dc_fields


@dataclass
class StrategyParams:

    # ── ENTRY: Lens 1 — VWAP Signal Quality ──────────────────────────────────
    threshold:                   float = 0.005  # core VWAP deviation to trigger signal (0.5 %)
    vwap_dev_max:                float = 0.0    # 0=disabled; skip if |dev| > this (over-extended)
    vwap_sustained_candles:      int   = 1      # consecutive candles away from VWAP required (≥1)

    # ── ENTRY: Lens 2 — Momentum Confirmation (RSI) ───────────────────────────
    rsi_oversold:                float = 100.0  # long only if RSI < this  (100 = disabled)
    rsi_overbought:              float = 0.0    # short only if RSI > this (0   = disabled)
    rsi_reversal_required:       bool  = False  # RSI must already be turning in trade direction

    # ── ENTRY: Lens 3 — Volume Confirmation ───────────────────────────────────
    volume_filter:               float = 1.5    # minimum volume vs rolling avg (1×–3×)
    volume_declining_on_move:    bool  = False  # require volume this candle < prev candle (exhaustion)

    # ── ENTRY: Lens 4 — Time / Session ────────────────────────────────────────
    time_open_filter:            int   = 15     # skip first N candles of the day
    time_close_filter:           int   = 15     # skip last N candles of the day
    preferred_session_start:     int   = 0      # 0=disabled; require minute_of_day >= this
    preferred_session_end:       int   = 0      # 0=disabled; require minute_of_day <= this

    # ── ENTRY: Lens 5 — Market Regime ─────────────────────────────────────────
    adx_trend_cap:               float = 0.0    # 0=disabled; skip entry if ADX14 > this value

    # ── ENTRY: Lens 6 — Position Management ───────────────────────────────────
    max_daily_entries:           int   = 0      # 0=unlimited; max new entries per trading day
    cooldown_after_loss_minutes: int   = 0      # 0=disabled; pause N minutes after a losing trade

    # ── EXIT: Lens 1 — Static Price ───────────────────────────────────────────
    stop_loss:                   float = 0.003  # fixed stop distance from entry price (0.3 %)
    take_profit:                 float = 0.006  # fixed take-profit distance from entry (0.6 %)

    # ── EXIT: Lens 2 — Dynamic Price ──────────────────────────────────────────
    trailing_stop_pct:           float = 0.0    # 0=disabled; exit when PnL falls this % from peak
    breakeven_trigger_pct:       float = 0.0    # 0=disabled; move stop to entry when trade reaches this %

    # ── EXIT: Lens 3 — Time ───────────────────────────────────────────────────
    max_holding:                 int   = 30     # force-exit after this many minutes (5–60)
    session_hard_exit_buffer:    int   = 0      # 0=disabled; force exit N extra candles before EOD zone

    # ── EXIT: Lens 4 — Regime Change ──────────────────────────────────────────
    volatility_exit_multiplier:  float = 0.0    # 0=disabled; exit if live ATR > N × entry ATR

    # ── EXIT: Lens 5 — Signal Invalidation ────────────────────────────────────
    signal_flip_exit_pct:        float = 0.0    # 0=disabled; exit if deviation extends beyond
                                                #   threshold + this (thesis failing fast)

    # ── EXIT: Lens 6 — Momentum Deterioration ─────────────────────────────────
    exit_rsi_cross:              bool  = False  # exit if RSI crosses 50 against trade direction

    # ── EXIT: Lens 7 — Partial Scaling ────────────────────────────────────────
    partial_exit_ratio:          float = 0.0    # 0=disabled; fraction of position to close early
    partial_exit_trigger_pct:    float = 0.5    # trigger partial at this fraction of take_profit

    # ── Costs ──────────────────────────────────────────────────────────────────
    slippage:                    float = 0.0001 # per-side slippage  (0.01 %)
    commission:                  float = 0.0003 # per-side commission (0.03 %)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# Bounds for hyperparameter search (min, max).
# Only core parameters are included here so the optimizer search space stays
# tractable.  The new lens parameters use their dataclass defaults during
# optimisation; users can tune them manually or extend PARAM_BOUNDS as needed.
PARAM_BOUNDS = {
    "threshold":         (0.002, 0.020),
    "volume_filter":     (1.0,   3.0),
    "stop_loss":         (0.001, 0.010),
    "take_profit":       (0.001, 0.020),
    "max_holding":       (5,     60),
    "time_open_filter":  (0,     60),
    "time_close_filter": (0,     60),
}

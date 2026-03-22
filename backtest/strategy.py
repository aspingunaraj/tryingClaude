"""5-Min Trend Pullback Engulfing Strategy — parameter definitions.

One intraday strategy on 5-minute candles:
  - Trend: close vs VWAP + close vs EMA(20) + higher-high/lower-low lookback
  - Pullback to VWAP/EMA zone before entry
  - Bullish or bearish engulfing candle at pullback zone
  - Volume confirmation: engulfing volume > avg of last N candles
  - Entry: close of engulfing candle (signal bar)
  - Stop: engulfing candle low (long) / high (short)
  - Exit: 1:2 RR (Option A)  or  trailing EMA20 stop (Option B)
  - Session: 9:30–13:30 IST only  (candle 3 – 51 within day, 9:15 open)
  - ATR sideways filter: skip if ATR/close < atr_sideways_pct
"""
from dataclasses import dataclass, fields as dc_fields


@dataclass
class StrategyParams:

    # ── Trend Detection ───────────────────────────────────────────────────────
    trend_lookback:          int   = 3        # candles back for HH/HL (or LL/LH) check

    # ── Pullback Detection ────────────────────────────────────────────────────
    pullback_zone_pct:       float = 0.005    # max distance from VWAP or EMA20 (fraction)

    # ── ATR Sideways Filter ───────────────────────────────────────────────────
    atr_sideways_pct:        float = 0.002    # skip trade if ATR/close < this

    # ── Volume Confirmation ───────────────────────────────────────────────────
    volume_lookback:         int   = 5        # periods for volume average comparison

    # ── Exit ─────────────────────────────────────────────────────────────────
    risk_reward_ratio:       float = 2.0      # TP = entry ± (entry − SL) × RR
    use_trailing_stop:       bool  = False    # False = fixed RR, True = trail EMA20
    max_holding_candles:     int   = 12       # force-exit after N 5-min candles (≈ 60 min)
    eod_buffer_candles:      int   = 6        # force-exit N candles before day end

    # ── Session (candle index within day, assuming 9:15 IST open on 5-min bars) ─
    session_start_candle:    int   = 3        # 9:30 IST = 3rd 5-min candle (index 0 = 9:15)
    session_end_candle:      int   = 51       # 13:30 IST = 51st candle

    # ── Transaction Costs (fixed — not optimised) ─────────────────────────────
    slippage:                float = 0.0001   # per-side slippage  (0.01 %)
    commission:              float = 0.0003   # per-side commission (0.03 %)

    # -------------------------------------------------------------------------

    def cost_per_side(self) -> float:
        return self.slippage + self.commission

    def round_trip_cost(self) -> float:
        return 2.0 * self.cost_per_side()

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ---------------------------------------------------------------------------
# Optimiser search space — logic params only (costs, session, bool are fixed)
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "trend_lookback":       (2,     8),
    "pullback_zone_pct":    (0.001, 0.01),
    "atr_sideways_pct":     (0.001, 0.005),
    "volume_lookback":      (3,     10),
    "risk_reward_ratio":    (1.5,   3.0),
    "max_holding_candles":  (6,     24),
    "eod_buffer_candles":   (3,     10),
}

INT_PARAMS: frozenset = frozenset({
    "trend_lookback",
    "volume_lookback",
    "max_holding_candles",
    "eod_buffer_candles",
})

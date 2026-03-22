"""Multi-Strategy Ensemble — parameter definitions.

Three intraday strategies, each active in its own regime:

  Strategy 1: VWAP Trend Pullback   — active in TREND regime
  Strategy 2: Opening Range Breakout — active in BREAKOUT regime
  Strategy 3: VWAP Rejection MR     — active in RANGE regime

Regime is detected per-symbol from its own OHLCV data; only one
strategy fires at a time.  Total optimisable parameters: 9.
Fixed cost params (slippage, commission) are not optimised.

PARAM_BOUNDS convention:
  float entries → suggest_float in Optuna
  int   entries → suggest_int  in Optuna  (listed in INT_PARAMS)
"""
from dataclasses import dataclass, fields as dc_fields


@dataclass
class StrategyParams:

    # ── Regime Detection ──────────────────────────────────────────────────────
    # TREND    : abs(vwap_slope) > vwap_slope_threshold
    # BREAKOUT : atr > regime_atr_multiplier × atr_avg
    # RANGE    : everything else
    vwap_slope_threshold:   float = 0.0003   # VWAP slope magnitude for TREND label
    regime_atr_multiplier:  float = 1.5      # ATR spike multiple for BREAKOUT label

    # ── Strategy 1: VWAP Trend Pullback ──────────────────────────────────────
    # Long  : uptrend + price above VWAP within pullback_distance + bullish candle
    # Short : downtrend + price below VWAP within pullback_distance + bearish candle
    pullback_distance:      float = 0.003    # max allowed (close-vwap)/vwap for pullback entry

    # ── Strategy 2: Opening Range Breakout ───────────────────────────────────
    opening_range_minutes:  int   = 15       # number of candles defining the opening range
    breakout_buffer:        float = 0.001    # price must exceed range by this fraction

    # ── Common Exit Parameters ────────────────────────────────────────────────
    atr_stop_multiplier:    float = 1.2      # stop distance = N × ATR at entry
    risk_reward_ratio:      float = 1.8      # take-profit = RR × stop distance
    max_holding_minutes:    int   = 30       # force-exit after N candles in position

    # ── EOD buffer ────────────────────────────────────────────────────────────
    eod_buffer_candles:     int   = 15       # force-exit this many candles before day end

    # ── Transaction Costs (fixed — not optimised) ─────────────────────────────
    slippage:               float = 0.0001   # per-side slippage  (0.01 %)
    commission:             float = 0.0003   # per-side commission (0.03 %)

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
# Optimiser search space — 9 parameters (costs are fixed, not optimised).
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    # Regime detection
    "vwap_slope_threshold":  (0.0001, 0.001),
    "regime_atr_multiplier": (1.2,    3.0),

    # Strategy 1
    "pullback_distance":     (0.001,  0.01),

    # Strategy 2
    "opening_range_minutes": (5,      30),
    "breakout_buffer":       (0.0005, 0.005),

    # Exit
    "atr_stop_multiplier":   (0.5,    2.5),
    "risk_reward_ratio":     (1.0,    3.0),
    "max_holding_minutes":   (10,     60),
    "eod_buffer_candles":    (5,      30),
}

# Parameters that must be sampled as integers
INT_PARAMS: frozenset = frozenset({
    "opening_range_minutes",
    "max_holding_minutes",
    "eod_buffer_candles",
})

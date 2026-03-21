"""Strategy parameters and parameter-space definitions.

StrategyParams is a plain dataclass — the backtester and optimizer both consume it.
PARAM_BOUNDS defines the search space for optimisation.
"""
from dataclasses import dataclass, fields as dc_fields


@dataclass
class StrategyParams:
    # --- Entry ---
    threshold:         float = 0.005  # VWAP deviation before entry (0.2 %–2 %)
    volume_filter:     float = 1.5    # Minimum volume vs rolling avg (1×–3×)

    # --- Exit ---
    stop_loss:         float = 0.003  # Stop-loss distance from entry price (0.1 %–1 %)
    take_profit:       float = 0.006  # Take-profit distance from entry price (0.1 %–2 %)
    max_holding:       int   = 30     # Force-exit after this many minutes (5–60)

    # --- Time filters ---
    time_open_filter:  int   = 15     # Skip first N candles of the day (0–60)
    time_close_filter: int   = 15     # Skip last N candles of the day (0–60)

    # --- Costs ---
    slippage:          float = 0.0001 # Per-side slippage (0.01 %)
    commission:        float = 0.0003 # Per-side commission (0.03 %)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# Bounds for hyperparameter search (min, max)
PARAM_BOUNDS = {
    "threshold":         (0.002, 0.020),
    "volume_filter":     (1.0,   3.0),
    "stop_loss":         (0.001, 0.010),
    "take_profit":       (0.001, 0.020),
    "max_holding":       (5,     60),
    "time_open_filter":  (0,     60),
    "time_close_filter": (0,     60),
}

"""
Dynamic position sizing and ML configuration.

PositionSizer maps a model probability → position size multiplier.
MLConfig bundles all ML-related hyper-parameters.

Default sizing brackets:
    prob < filter_threshold  → no trade  (size = 0.0)
    0.50 – 0.60              → 0.5×
    0.60 – 0.70              → 1.0×
    0.70 – 0.80              → 1.5×
    ≥ 0.80                   → 2.0×
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Position sizer
# ---------------------------------------------------------------------------

@dataclass
class PositionSizer:
    """
    Map an ML probability score to a position size multiplier.

    brackets : list of (prob_low_inclusive, prob_high_exclusive, size)
               Must be ordered; the first matching bracket wins.
    max_size  : hard cap — no bracket can exceed this.
    """
    brackets: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.50, 0.60, 0.5),
        (0.60, 0.70, 1.0),
        (0.70, 0.80, 1.5),
        (0.80, 1.01, 2.0),
    ])
    max_size: float = 2.0

    def get_size(self, prob: float) -> float:
        """
        Return the position size multiplier for a given probability.
        Returns 0.0 if prob falls below the lowest bracket (= no trade).
        """
        for lo, hi, size in self.brackets:
            if lo <= prob < hi:
                return min(size, self.max_size)
        return 0.0


# ---------------------------------------------------------------------------
# ML configuration
# ---------------------------------------------------------------------------

@dataclass
class MLConfig:
    """
    All toggles and thresholds for the ML-enhanced backtest pass.

    Attributes
    ----------
    enabled              : master switch — if False, run_backtest_ml behaves
                           like run_backtest (no filtering, size=1)
    filter_threshold     : minimum predicted probability to open a trade
                           (trades below this are skipped entirely)
    regime_filter        : if True, skip signals in TRENDING regime
    adx_threshold        : ADX below this value → mean-reverting regime
    ema_slope_threshold  : |ema_slope| below this → flat trend → MR regime
    """
    enabled:             bool  = True
    filter_threshold:    float = 0.55
    regime_filter:       bool  = True
    adx_threshold:       float = 25.0
    ema_slope_threshold: float = 3e-4

    def to_dict(self) -> dict:
        return {
            "enabled":             self.enabled,
            "filter_threshold":    self.filter_threshold,
            "regime_filter":       self.regime_filter,
            "adx_threshold":       self.adx_threshold,
            "ema_slope_threshold": self.ema_slope_threshold,
        }

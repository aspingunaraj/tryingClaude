"""Performance metrics for a completed backtest pass."""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def compute_metrics(trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict:
    """
    Compute standard strategy performance metrics.

    Parameters
    ----------
    trades_df    : DataFrame returned by backtester.run_backtest
    equity_curve : Series returned by backtester.run_backtest

    Returns
    -------
    dict with:
      total_return_pct, sharpe_ratio, max_drawdown_pct,
      win_rate_pct, profit_factor, avg_trade_pct, n_trades
    """
    if trades_df is None or trades_df.empty:
        return _empty_metrics()

    pnls = trades_df["pnl_pct"].dropna()
    n    = len(pnls)
    if n == 0:
        return _empty_metrics()

    total_return = float(pnls.sum())

    wins         = (pnls > 0).sum()
    win_rate     = wins / n

    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss   = float(abs(pnls[pnls < 0].sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_trade = float(pnls.mean())

    # Sharpe based on daily PnL (252 trading days annualisation)
    trades_copy = trades_df.copy()
    trades_copy["date"] = pd.to_datetime(trades_copy["exit_time"]).dt.date
    daily_pnl   = trades_copy.groupby("date")["pnl_pct"].sum()
    daily_std   = float(daily_pnl.std())
    sharpe      = (float(daily_pnl.mean()) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    max_dd = _max_drawdown(equity_curve)

    return {
        "total_return_pct": _r(total_return * 100, 4),
        "sharpe_ratio":     _r(sharpe, 4),
        "max_drawdown_pct": _r(max_dd  * 100, 4),
        "win_rate_pct":     _r(win_rate * 100, 2),
        "profit_factor":    _r(profit_factor, 4),
        "avg_trade_pct":    _r(avg_trade * 100, 4),
        "n_trades":         int(n),
    }


def objective_score(metrics: Dict) -> float:
    """
    Composite optimisation objective (higher = better).

    Rewards  : Sharpe ratio
    Penalises: max drawdown (2×), low trade count (< 20)
    """
    n      = metrics["n_trades"]
    sharpe = metrics["sharpe_ratio"]
    mdd    = metrics["max_drawdown_pct"] / 100.0

    if n < 10:
        return -999.0   # not enough trades to be meaningful

    few_trade_penalty = max(0.0, (20 - n) / 20.0)
    return sharpe - 2.0 * mdd - few_trade_penalty


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown    = equity_curve - running_max
    return float(abs(drawdown.min()))


def _empty_metrics() -> Dict:
    return {
        "total_return_pct": 0.0,
        "sharpe_ratio":     0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct":     0.0,
        "profit_factor":    0.0,
        "avg_trade_pct":    0.0,
        "n_trades":         0,
    }


def aggregate_across_stocks(per_stock_results: list) -> Dict:
    """
    Aggregate metrics across multiple stocks.

    Parameters
    ----------
    per_stock_results : list of dicts, each containing:
        {
          "symbol":       str,
          "train_metrics": dict,  (from compute_metrics)
          "test_metrics":  dict,
          "test_trades":   pd.DataFrame,
          "test_equity":   pd.Series,
        }

    Returns
    -------
    {
      "portfolio_metrics": dict  – computed on pooled test trades,
      "avg_metrics":       dict  – simple per-metric mean across stocks,
      "n_stocks":          int,
      "n_stocks_profitable": int,
    }
    """
    if not per_stock_results:
        return {
            "portfolio_metrics": _empty_metrics(),
            "avg_metrics":       _empty_metrics(),
            "n_stocks":          0,
            "n_stocks_profitable": 0,
        }

    # Pool all test trades
    all_trades = pd.concat(
        [r["test_trades"] for r in per_stock_results if not r["test_trades"].empty],
        ignore_index=True,
    )

    # Pool equity curves (sum = equal-weight portfolio)
    eq_list = [r["test_equity"].reset_index(drop=True)
               for r in per_stock_results if r["test_equity"] is not None]
    if eq_list:
        min_len      = min(len(e) for e in eq_list)
        portfolio_eq = sum(e.iloc[:min_len] for e in eq_list)
    else:
        portfolio_eq = pd.Series([], dtype=float)

    portfolio_metrics = compute_metrics(all_trades, portfolio_eq)

    # Per-metric averages
    metric_keys = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                   "win_rate_pct", "profit_factor", "avg_trade_pct", "n_trades"]
    avg_metrics = {}
    for k in metric_keys:
        vals = [r["test_metrics"].get(k, 0.0) for r in per_stock_results]
        avg  = float(np.mean([v for v in vals if v is not None]))
        avg_metrics[k] = _r(avg if k != "n_trades" else round(avg), 4 if k != "n_trades" else 0)

    n_profitable = sum(
        1 for r in per_stock_results
        if r["test_metrics"].get("total_return_pct", 0) > 0
    )

    return {
        "portfolio_metrics":    portfolio_metrics,
        "avg_metrics":          avg_metrics,
        "n_stocks":             len(per_stock_results),
        "n_stocks_profitable":  n_profitable,
    }


def _r(v, d: int):
    """Round, handling inf gracefully."""
    if v == float("inf"):
        return 9999.0
    return round(float(v), d)

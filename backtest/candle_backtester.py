"""
Candlestick pattern backtester.

Trade mechanics
---------------
* Entry  : open of the candle AFTER the signal fires
* SL     : entry ± sl_pct  (long: entry*(1-sl), short: entry*(1+sl))
* TP     : entry ± tp_pct  (long: entry*(1+tp), short: entry*(1-tp))
* Size   : always 1 share → P&L in ₹
* Intrabar exit: on each subsequent candle check high/low for SL/TP breach
                 (if both breached in same candle → SL wins, conservative)
* EOD exit: any open trade is closed at the day's last candle close
* One trade at a time — new signals ignored while a trade is open
"""
from __future__ import annotations

import io
import base64
from typing import Optional

import numpy as np
import pandas as pd

from backtest.data_loader import load_csv, resample_to_5min
from backtest.candle_patterns import detect_all, PATTERN_LABELS


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

def run_candle_backtest(
    df5: pd.DataFrame,          # 5-min OHLCV with 'date' column
    sl_pct:           float,    # e.g. 0.01 = 1%
    tp_pct:           float,    # e.g. 0.02 = 2%
    max_hold_candles: int,      # close trade if neither SL/TP hit after N candles
    enabled_patterns: list[str],
) -> dict:
    """
    Run the candlestick pattern strategy on a pre-resampled 5-min DataFrame.

    Returns
    -------
    {
      "trades":       pd.DataFrame,
      "equity_curve": pd.Series,   (cumulative ₹ P&L, indexed by trade number)
    }
    """
    df = detect_all(df5, enabled_patterns).reset_index(drop=True)
    n  = len(df)

    trades = []
    in_trade   = False
    entry_idx  = None
    entry_price = None
    direction  = None
    sl_price   = None
    tp_price   = None
    pattern_name = None

    for i in range(n):
        row = df.iloc[i]

        # ── Manage open trade ─────────────────────────────────────────────
        if in_trade:
            hi, lo, close_p = row["high"], row["low"], row["close"]
            exit_price = None
            exit_reason = None
            candles_held = i - entry_idx

            if direction == "long":
                sl_hit = lo  <= sl_price
                tp_hit = hi  >= tp_price
                if sl_hit and (not tp_hit or lo <= sl_price):
                    exit_price, exit_reason = sl_price, "sl"
                elif tp_hit:
                    exit_price, exit_reason = tp_price, "tp"
            else:  # short
                sl_hit = hi >= sl_price
                tp_hit = lo <= tp_price
                if sl_hit and (not tp_hit or hi >= sl_price):
                    exit_price, exit_reason = sl_price, "sl"
                elif tp_hit:
                    exit_price, exit_reason = tp_price, "tp"

            # Max-hold or EOD
            is_last_of_day = (
                i == n - 1 or df.iloc[i + 1]["date"] != row["date"]
            )
            if exit_price is None and (candles_held >= max_hold_candles or is_last_of_day):
                exit_price  = close_p
                exit_reason = "eod" if is_last_of_day else "timeout"

            if exit_price is not None:
                pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
                pnl_pct = pnl / entry_price

                trades.append({
                    "entry_time":   df.iloc[entry_idx]["datetime"],
                    "exit_time":    row["datetime"],
                    "direction":    direction,
                    "pattern":      pattern_name,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "sl_price":     sl_price,
                    "tp_price":     tp_price,
                    "pnl":          round(pnl, 4),
                    "pnl_pct":      round(pnl_pct, 6),
                    "exit_reason":  exit_reason,
                })
                in_trade = False

        # ── Check for new signal (only when flat) ─────────────────────────
        if not in_trade and row["signal"] is not None:
            # Entry is next candle's open
            if i + 1 < n and df.iloc[i + 1]["date"] == row["date"]:
                next_row    = df.iloc[i + 1]
                entry_price = next_row["open"]
                direction   = row["signal"]
                pattern_name = row["pattern"]
                entry_idx   = i + 1

                if direction == "long":
                    sl_price = entry_price * (1 - sl_pct)
                    tp_price = entry_price * (1 + tp_pct)
                else:
                    sl_price = entry_price * (1 + sl_pct)
                    tp_price = entry_price * (1 - tp_pct)

                in_trade = True

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "entry_time", "exit_time", "direction", "pattern",
        "entry_price", "exit_price", "sl_price", "tp_price",
        "pnl", "pnl_pct", "exit_reason",
    ])

    # Equity curve: cumulative ₹ P&L (1 share)
    if not trades_df.empty:
        equity = trades_df["pnl"].cumsum()
        equity.index = range(len(equity))
    else:
        equity = pd.Series(dtype=float)

    return {"trades": trades_df, "equity_curve": equity}


# ---------------------------------------------------------------------------
# Metrics (₹-based since size = 1 share)
# ---------------------------------------------------------------------------

def compute_candle_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return _empty()

    pnls = trades_df["pnl"].dropna()
    n    = len(pnls)
    if n == 0:
        return _empty()

    wins          = int((pnls > 0).sum())
    win_rate      = wins / n
    total_pnl     = float(pnls.sum())
    avg_pnl       = float(pnls.mean())
    gross_profit  = float(pnls[pnls > 0].sum())
    gross_loss    = float(abs(pnls[pnls < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe on daily ₹ P&L
    tmp = trades_df.copy()
    tmp["date"] = pd.to_datetime(tmp["exit_time"]).dt.date
    daily = tmp.groupby("date")["pnl"].sum()
    std   = float(daily.std())
    sharpe = (float(daily.mean()) / std * np.sqrt(252)) if std > 0 else 0.0

    # Max drawdown on cumulative ₹ P&L
    cum = pnls.cumsum()
    dd  = float((cum - cum.cummax()).min())

    # Breakdown by pattern
    pattern_counts = trades_df.groupby("pattern")["pnl"].agg(
        trades="count", total_pnl="sum", win_rate=lambda x: (x > 0).mean() * 100
    ).round(2).to_dict("index")

    return {
        "n_trades":       int(n),
        "win_rate_pct":   round(win_rate * 100, 2),
        "total_pnl":      round(total_pnl, 2),
        "avg_pnl":        round(avg_pnl, 2),
        "profit_factor":  round(profit_factor, 4) if profit_factor != float("inf") else 999,
        "sharpe_ratio":   round(sharpe, 4),
        "max_drawdown":   round(dd, 2),
        "pattern_breakdown": pattern_counts,
    }


def _empty() -> dict:
    return {
        "n_trades":          0,
        "win_rate_pct":      0.0,
        "total_pnl":         0.0,
        "avg_pnl":           0.0,
        "profit_factor":     0.0,
        "sharpe_ratio":      0.0,
        "max_drawdown":      0.0,
        "pattern_breakdown": {},
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def generate_candle_chart(
    equity: pd.Series,
    trades: pd.DataFrame,
    symbol: str,
) -> str:
    """Return a base64-encoded PNG: equity curve + trade breakdown bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(10, 5), facecolor="#1a1a2e")
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # ── Equity curve ──────────────────────────────────────────────────
        ax1.set_facecolor("#0f0f1a")
        if not equity.empty:
            color = "#22c55e" if float(equity.iloc[-1]) >= 0 else "#ef4444"
            ax1.plot(equity.values, color=color, linewidth=1.2)
            ax1.axhline(0, color="#666", linewidth=0.5, linestyle="--")
            ax1.fill_between(range(len(equity)), equity.values, 0,
                             alpha=0.15, color=color)
        ax1.set_title(f"{symbol} — Equity (₹)", color="#e2e8f0", fontsize=9)
        ax1.tick_params(colors="#888", labelsize=7)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#333")

        # ── Pattern breakdown bar chart ───────────────────────────────────
        ax2.set_facecolor("#0f0f1a")
        if trades is not None and not trades.empty:
            by_pat = trades.groupby("pattern")["pnl"].sum().sort_values()
            labels = [PATTERN_LABELS.get(p, p) for p in by_pat.index]
            colors = ["#22c55e" if v >= 0 else "#ef4444" for v in by_pat.values]
            bars   = ax2.barh(labels, by_pat.values, color=colors, height=0.5)
            ax2.axvline(0, color="#666", linewidth=0.5, linestyle="--")
            for bar, val in zip(bars, by_pat.values):
                ax2.text(val + (0.5 if val >= 0 else -0.5),
                         bar.get_y() + bar.get_height() / 2,
                         f"₹{val:.0f}", va="center", ha="left" if val >= 0 else "right",
                         color="#e2e8f0", fontsize=6.5)
        ax2.set_title("P&L by Pattern (₹)", color="#e2e8f0", fontsize=9)
        ax2.tick_params(colors="#888", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Pipeline — runs across all configured stocks
# ---------------------------------------------------------------------------

def run_candle_pipeline(
    stocks_cfg:       list,
    sl_pct:           float,
    tp_pct:           float,
    max_hold_candles: int,
    enabled_patterns: list[str],
) -> dict:
    """
    Load each stock, resample to 5-min, run the candle pattern backtest.

    Returns
    -------
    {
      "per_stock":  [ {symbol, exchange, metrics, chart_b64}, … ],
      "n_configured": int,
      "n_loaded":   int,
    }
    """
    print(f"\n{'='*60}")
    print(f"  Candlestick Patterns  |  {len(stocks_cfg)} stocks")
    print(f"  Patterns: {enabled_patterns}")
    print(f"  SL={sl_pct*100:.1f}%  TP={tp_pct*100:.1f}%  MaxHold={max_hold_candles}")
    print(f"{'='*60}")

    per_stock = []
    for s in stocks_cfg:
        sym, exch = s["symbol"], s["exchange"]
        try:
            raw = load_csv(sym, exch)
            df5 = resample_to_5min(raw)
            result  = run_candle_backtest(df5, sl_pct, tp_pct, max_hold_candles, enabled_patterns)
            metrics = compute_candle_metrics(result["trades"])
            chart   = generate_candle_chart(result["equity_curve"], result["trades"], sym)
            print(
                f"  {sym:<12s}  trades={metrics['n_trades']}"
                f"  WR={metrics['win_rate_pct']:.1f}%"
                f"  PnL=₹{metrics['total_pnl']:.0f}"
                f"  Sh={metrics['sharpe_ratio']:.2f}"
            )
            per_stock.append({
                "symbol":   sym,
                "exchange": exch,
                "metrics":  metrics,
                "chart_b64": chart,
            })
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  ERROR {sym}: {exc}")
            per_stock.append({
                "symbol":   sym,
                "exchange": exch,
                "metrics":  _empty(),
                "chart_b64": "",
                "error":    str(exc),
            })

    return {
        "per_stock":    per_stock,
        "n_configured": len(stocks_cfg),
        "n_loaded":     sum(1 for r in per_stock if "error" not in r),
    }

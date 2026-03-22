"""
Full backtesting pipeline — CLI entry point.

Usage examples
--------------
# Quick run with default parameters (no optimisation):
  python -m backtest.main --symbol INFY

# Run with Bayesian optimisation (100 trials):
  python -m backtest.main --symbol INFY --optimize --trials 100

# Run all configured stocks:
  python -m backtest.main --all

# Walk-forward validation:
  python -m backtest.main --symbol INFY --walkforward
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import base64
import io

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data_loader  import load_csv, resample_to_5min, split_train_test
from backtest.indicators   import add_all_indicators
from backtest.strategy     import StrategyParams
from backtest.backtester   import run_backtest
from backtest.metrics      import compute_metrics, aggregate_across_stocks
from backtest.optimizer    import optimize, optimize_cross_stock, walk_forward

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min data to 5-min and add indicators."""
    df5 = resample_to_5min(df)
    return add_all_indicators(df5)


def _print_section(title: str, data: dict) -> None:
    print(f"\n── {title} {'─'*(50-len(title))}")
    for k, v in data.items():
        print(f"  {k:<30s}: {v}")


def _print_overfitting_check(train: dict, test: dict) -> None:
    ts = train.get("sharpe_ratio", 0)
    ss = test.get("sharpe_ratio", 0)
    ratio = abs(ss / ts) if ts != 0 else 0
    flag  = "⚠ Possible overfit" if ratio < 0.5 else "✓ OK"
    print(f"\n  Sharpe ratio train={ts:.2f}  test={ss:.2f}  ratio={ratio:.2f}  {flag}")


# ---------------------------------------------------------------------------
# Single-symbol pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    symbol:          str,
    exchange:        str  = "NSE",
    optimize_params: bool = False,
    n_trials:        int  = 100,
    default_params:  dict = None,
    walk_fwd:        bool = False,
) -> dict:
    """
    Load → resample → split → (optimise on train) → evaluate on test → save results.
    Returns a summary dict suitable for JSON serialisation.
    """
    print(f"\n{'='*60}")
    print(f"  5-Min Trend Pullback Engulfing  |  {exchange}:{symbol}")
    print(f"{'='*60}")

    # 1. Load (1-min) and split by date before resampling
    df = load_csv(symbol, exchange)
    print(f"Loaded {len(df):,} 1-min candles across {df['date'].nunique()} trading days")

    train_df, test_df, train_days, test_days = split_train_test(df)
    print(f"Train : {train_days[0]} → {train_days[-1]}  ({len(train_days)} days)")
    print(f"Test  : {test_days[0]} → {test_days[-1]}   ({len(test_days)} days)")

    # 2. Resample + indicators
    train_prep = _prepare(train_df)
    test_prep  = _prepare(test_df)

    # 3. Optimise or use provided / default params
    if optimize_params:
        print(f"\nBayesian optimisation ({n_trials} trials)…")
        best_params, train_metrics = optimize(train_df, n_trials=n_trials,
                                              show_progress=True)
    else:
        best_params   = StrategyParams.from_dict(default_params or {})
        train_result  = run_backtest(train_prep, best_params)
        train_metrics = compute_metrics(train_result["trades"],
                                        train_result["equity_curve"])

    _print_section("Best Parameters", best_params.to_dict())
    _print_section("Train Metrics",   train_metrics)

    # 4. Test set evaluation
    print("\nEvaluating on test data…")
    test_result  = run_backtest(test_prep, best_params)
    test_metrics = compute_metrics(test_result["trades"], test_result["equity_curve"])

    _print_section("Test Metrics", test_metrics)
    _print_overfitting_check(train_metrics, test_metrics)

    # 5. Walk-forward (optional)
    wf_results = []
    if walk_fwd:
        print("\nWalk-forward validation…")
        wf_results = walk_forward(df, n_windows=5, train_days=20,
                                  test_days=10, n_trials=30)

    # 6. Re-run train for full trade log + charts
    train_result_full = run_backtest(train_prep, best_params)

    # 7. Save trade CSVs
    tag = f"{exchange}_{symbol}"
    train_result_full["trades"].to_csv(
        os.path.join(RESULTS_DIR, f"{tag}_train_trades.csv"), index=False)
    test_result["trades"].to_csv(
        os.path.join(RESULTS_DIR, f"{tag}_test_trades.csv"), index=False)

    # 8. Charts
    chart_path, chart_b64 = generate_charts(
        train_result_full["equity_curve"], test_result["equity_curve"],
        train_result_full["trades"],       test_result["trades"],
        symbol, exchange,
    )

    # 9. Save summary JSON
    summary = {
        "symbol":        symbol,
        "exchange":      exchange,
        "best_params":   best_params.to_dict(),
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "chart_path":    chart_path,
        "chart_b64":     chart_b64,
        "wf_results":    wf_results,
        "last_signal":   test_result.get("last_signal"),
    }
    with open(os.path.join(RESULTS_DIR, f"{tag}_summary.json"), "w") as f:
        out = {k: v for k, v in summary.items() if k != "chart_b64"}
        json.dump(out, f, indent=2, default=str)

    print(f"\nResults saved → {RESULTS_DIR}/")
    return summary


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(
    train_eq, test_eq,
    train_trades, test_trades,
    symbol, exchange,
) -> tuple[str, str]:
    """Build a 3-row dashboard. Returns (file_path, base64_png).
    Returns ('', '') if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return "", ""

    C_TRAIN = "#dc2626"
    C_TEST  = "#22c55e"
    BG      = "#181818"
    GRID    = "#2a2a2a"
    TEXT    = "#cccccc"

    fig = plt.figure(figsize=(14, 10), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors=TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # 1 ── Equity curve (full width)
    ax1 = fig.add_subplot(gs[0, :])
    offset = len(train_eq)
    ax1.plot(range(offset), train_eq.values,
             color=C_TRAIN, lw=1.4, label="Train")
    last_train = float(train_eq.iloc[-1]) if len(train_eq) else 0
    ax1.plot(range(offset, offset + len(test_eq)),
             test_eq.values + last_train,
             color=C_TEST, lw=1.4, label="Test")
    ax1.axvline(offset, color="#555", lw=1, ls="--", label="Split")
    ax1.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax1, "Equity Curve — cumulative fractional PnL")

    # 2 ── Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    for eq, color, lbl in [(train_eq, C_TRAIN, "Train"), (test_eq, C_TEST, "Test")]:
        dd = eq - eq.cummax()
        ax2.fill_between(range(len(dd)), dd.values, 0, color=color, alpha=0.4, label=lbl)
    ax2.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax2, "Drawdown")

    # 3 ── Return distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if not train_trades.empty:
        ax3.hist(train_trades["pnl_pct"] * 100, bins=30,
                 color=C_TRAIN, alpha=0.6, label="Train")
    if not test_trades.empty:
        ax3.hist(test_trades["pnl_pct"] * 100, bins=30,
                 color=C_TEST,  alpha=0.6, label="Test")
    ax3.axvline(0, color="#888", lw=1, ls="--")
    ax3.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax3, "Trade Return Distribution (%)")

    # 4 ── Exit reason breakdown
    ax4 = fig.add_subplot(gs[2, 0])
    for trades, color, lbl in [
        (train_trades, C_TRAIN, "Tr"),
        (test_trades,  C_TEST,  "Te"),
    ]:
        if not trades.empty and "exit_reason" in trades.columns:
            counts = trades["exit_reason"].value_counts()
            bars   = [f"{r}\n({lbl})" for r in counts.index]
            ax4.bar(bars, counts.values, color=color, alpha=0.75)
    ax4.tick_params(axis="x", labelsize=6)
    style(ax4, "Exit Reason Breakdown")

    # 5 ── Holding time (in 5-min candles)
    ax5 = fig.add_subplot(gs[2, 1])
    if not train_trades.empty and "holding_minutes" in train_trades.columns:
        ax5.hist(train_trades["holding_minutes"], bins=20,
                 color=C_TRAIN, alpha=0.6, label="Train")
    if not test_trades.empty and "holding_minutes" in test_trades.columns:
        ax5.hist(test_trades["holding_minutes"], bins=20,
                 color=C_TEST, alpha=0.6, label="Test")
    ax5.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax5, "Holding Time (minutes)")

    fig.suptitle(
        f"5-Min Trend Pullback Engulfing  |  {exchange}:{symbol}",
        color=TEXT, fontsize=12, fontweight="bold", y=0.99,
    )

    chart_path = os.path.join(RESULTS_DIR, f"{exchange}_{symbol}_charts.png")
    plt.savefig(chart_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()

    return chart_path, chart_b64


# ---------------------------------------------------------------------------
# Combined multi-stock chart
# ---------------------------------------------------------------------------

def generate_combined_chart(
    per_stock_results: list,
    title: str = "Cross-Stock Results  |  5-Min Trend Pullback Engulfing",
) -> str:
    """Return base64 PNG of combined equity + win-rate chart.
    Returns '' if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return ""

    BG   = "#181818"
    GRID = "#2a2a2a"
    TEXT = "#cccccc"
    COLS = [
        "#22c55e", "#3b82f6", "#f59e0b", "#a855f7",
        "#ec4899", "#06b6d4", "#ef4444", "#84cc16",
    ]

    fig = plt.figure(figsize=(14, 7), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    def style(ax, t):
        ax.set_facecolor(BG)
        ax.set_title(t, color=TEXT, fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for i, r in enumerate(per_stock_results):
        eq = r.get("test_equity")
        if eq is None or (hasattr(eq, "empty") and eq.empty):
            continue
        color = COLS[i % len(COLS)]
        ax1.plot(eq.values, color=color, lw=1.2, alpha=0.8, label=r["symbol"])

    ax1.axhline(0, color="#555", lw=0.8, ls="--")
    ax1.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
               ncol=2, loc="upper left")
    style(ax1, "Test Equity Curves")

    # Win-rate bar chart
    syms  = [r["symbol"] for r in per_stock_results]
    wrs   = [r.get("test_metrics", {}).get("win_rate_pct", 0) for r in per_stock_results]
    bar_colors = [COLS[i % len(COLS)] for i in range(len(syms))]
    bars = ax2.bar(syms, wrs, color=bar_colors, alpha=0.85)
    for bar, wr in zip(bars, wrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{wr:.1f}%", ha="center", va="bottom", fontsize=7, color=TEXT)
    ax2.axhline(50, color="#f59e0b", lw=1, ls="--", label="50%")
    ax2.tick_params(axis="x", rotation=30, labelsize=7)
    style(ax2, "Win Rate by Stock (%)")

    fig.suptitle(title, color=TEXT, fontsize=11, fontweight="bold", y=1.02)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    return b64


# ---------------------------------------------------------------------------
# Multi-stock pipeline
# ---------------------------------------------------------------------------

def run_all_pipeline(
    stocks_cfg:      list,
    optimize_params: bool  = False,
    n_trials:        int   = 100,
    default_params:  dict  = None,
) -> dict:
    """
    Run backtest across ALL stocks with per-stock StrategyParams.

    When optimize_params=True each stock gets its own Bayesian optimisation
    on its train window.  When False, default_params are used for every stock.

    Returns
    -------
    {
      "per_stock":          [ {symbol, exchange, best_params, train_metrics,
                               test_metrics, chart_b64, last_signal}, … ],
      "aggregate":          {portfolio_metrics, avg_metrics, …},
      "combined_chart_b64": str,
      "n_configured":       int,
      "n_loaded":           int,
    }
    """
    print(f"\n{'='*60}")
    print(f"  5-Min Trend Pullback Engulfing  |  {len(stocks_cfg)} stocks")
    print(f"{'='*60}")

    # 1. Load, resample, split every stock
    loaded = []
    for s in stocks_cfg:
        try:
            df = load_csv(s["symbol"], s["exchange"])
            train_df, test_df, train_days, test_days = split_train_test(df)
            loaded.append({
                "symbol":     s["symbol"],
                "exchange":   s["exchange"],
                "raw_train":  train_df,
                "raw_test":   test_df,
                "train_prep": _prepare(train_df),
                "test_prep":  _prepare(test_df),
                "train_days": train_days,
                "test_days":  test_days,
            })
            print(f"  Loaded {s['symbol']}: {len(df):,} 1-min candles")
        except Exception as exc:
            print(f"  SKIP {s['symbol']}: {exc}")

    n_configured = len(stocks_cfg)
    n_loaded     = len(loaded)
    if n_loaded < n_configured:
        skipped = [s["symbol"] for s in stocks_cfg
                   if not any(l["symbol"] == s["symbol"] for l in loaded)]
        print(f"  Skipped {n_configured - n_loaded} stocks (no CSV): {skipped}")

    if not loaded:
        return {"error": "No stock data available. Fetch data first."}

    # 2. Per-stock optimisation / backtest
    fallback_params = StrategyParams.from_dict(default_params or {})
    per_stock_results = []

    for s in loaded:
        sym, exch = s["symbol"], s["exchange"]
        try:
            # Determine params for this stock
            if optimize_params:
                print(f"\n  Optimising {sym} ({n_trials} trials)…")
                stock_params, _score = optimize(
                    s["raw_train"],
                    n_trials=n_trials,
                    show_progress=False,
                )
                # _score may be a float (optuna) or a metrics dict (random search)
                score_val = _score if isinstance(_score, (int, float)) else _score.get("sharpe_ratio", 0)
                print(f"    train score: {score_val:.4f}")
            else:
                stock_params = fallback_params

            train_result  = run_backtest(s["train_prep"], stock_params)
            train_metrics = compute_metrics(train_result["trades"],
                                            train_result["equity_curve"])

            test_result   = run_backtest(s["test_prep"], stock_params)
            test_metrics  = compute_metrics(test_result["trades"],
                                            test_result["equity_curve"])

            last_signal = test_result.get("last_signal")
            if last_signal:
                last_signal["symbol"] = sym

            print(
                f"  {sym:<12s}"
                f"  Sharpe={test_metrics['sharpe_ratio']:.2f}"
                f"  WR={test_metrics['win_rate_pct']:.1f}%"
                f"  trades={test_metrics['n_trades']}"
            )

            _, chart_b64 = generate_charts(
                train_result["equity_curve"], test_result["equity_curve"],
                train_result["trades"],       test_result["trades"],
                sym, exch,
            )

            per_stock_results.append({
                "symbol":        sym,
                "exchange":      exch,
                "best_params":   stock_params.to_dict(),
                "train_metrics": train_metrics,
                "test_metrics":  test_metrics,
                "chart_b64":     chart_b64,
                "last_signal":   last_signal,
                # kept for aggregation (removed before final JSON)
                "test_trades":   test_result["trades"],
                "test_equity":   test_result["equity_curve"],
            })

        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  ERROR {sym}: {exc}")
            per_stock_results.append({
                "symbol":        sym,
                "exchange":      exch,
                "best_params":   {},
                "train_metrics": {},
                "test_metrics":  {},
                "chart_b64":     "",
                "last_signal":   None,
                "test_trades":   pd.DataFrame(),
                "test_equity":   pd.Series(dtype=float),
                "error":         str(exc),
            })

    # 4. Aggregate
    aggregate = aggregate_across_stocks(per_stock_results)

    # 5. Combined chart
    combined_b64 = generate_combined_chart(per_stock_results)

    # 6. Save per-stock params JSON
    per_stock_params = {r["symbol"]: r["best_params"] for r in per_stock_results}
    with open(os.path.join(RESULTS_DIR, "PER_STOCK_params.json"), "w") as f:
        json.dump(per_stock_params, f, indent=2)

    # Strip DataFrames before returning
    for r in per_stock_results:
        r.pop("test_trades", None)
        r.pop("test_equity", None)

    return {
        "per_stock":          per_stock_results,
        "aggregate":          aggregate,
        "combined_chart_b64": combined_b64,
        "n_configured":       n_configured,
        "n_loaded":           n_loaded,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_section(title: str, data: dict) -> None:
    print(f"\n── {title} {'─'*(50-len(title))}")
    for k, v in data.items():
        print(f"  {k:<30s}: {v}")


def _print_overfitting_check(train: dict, test: dict) -> None:
    ts = train.get("sharpe_ratio", 0)
    ss = test.get("sharpe_ratio", 0)
    ratio = abs(ss / ts) if ts != 0 else 0
    flag  = "Possible overfit" if ratio < 0.5 else "OK"
    print(f"\n  Sharpe train={ts:.2f}  test={ss:.2f}  ratio={ratio:.2f}  [{flag}]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Min Trend Pullback Engulfing Backtest")
    parser.add_argument("--symbol",      default="INFY")
    parser.add_argument("--exchange",    default="NSE")
    parser.add_argument("--optimize",    action="store_true")
    parser.add_argument("--trials",      type=int, default=100)
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--all",         action="store_true")
    args = parser.parse_args()

    if args.all:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "backtest_stocks.json")
        with open(cfg_path) as f:
            stocks_cfg = json.load(f).get("stocks", [])
        run_all_pipeline(stocks_cfg, optimize_params=args.optimize,
                         n_trials=args.trials)
    else:
        run_full_pipeline(
            symbol          = args.symbol,
            exchange        = args.exchange,
            optimize_params = args.optimize,
            n_trials        = args.trials,
            walk_fwd        = args.walkforward,
        )

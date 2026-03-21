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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Allow running as `python -m backtest.main` from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data_loader  import load_csv, split_train_test
from backtest.indicators   import add_all_indicators
from backtest.strategy     import StrategyParams
from backtest.backtester   import run_backtest
from backtest.metrics      import compute_metrics
from backtest.optimizer    import optimize, walk_forward

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Full pipeline
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
    Load → split → (optimise on train) → evaluate on test → save results.
    Returns a summary dict suitable for JSON serialisation.
    """
    print(f"\n{'='*60}")
    print(f"  VWAP Mean Reversion  |  {exchange}:{symbol}")
    print(f"{'='*60}")

    # 1. Load
    df = load_csv(symbol, exchange)
    print(f"Loaded {len(df):,} candles across {df['date'].nunique()} trading days")

    # 2. Split
    train_df, test_df, train_days, test_days = split_train_test(df)
    print(f"Train : {train_days[0]} → {train_days[-1]}  ({len(train_days)} days)")
    print(f"Test  : {test_days[0]} → {test_days[-1]}   ({len(test_days)} days)")

    # 3. Optimise or use provided / default params
    if optimize_params:
        print(f"\nBayesian optimisation ({n_trials} trials)…")
        best_params, train_metrics = optimize(train_df, n_trials=n_trials, show_progress=True)
    else:
        best_params   = StrategyParams.from_dict(default_params or {})
        train_prep    = add_all_indicators(train_df)
        train_result  = run_backtest(train_prep, best_params)
        train_metrics = compute_metrics(train_result["trades"], train_result["equity_curve"])

    _print_section("Best Parameters",  best_params.to_dict())
    _print_section("Train Metrics",    train_metrics)

    # 4. Test set evaluation
    print("\nEvaluating on test data…")
    test_prep    = add_all_indicators(test_df)
    test_result  = run_backtest(test_prep, best_params)
    test_metrics = compute_metrics(test_result["trades"], test_result["equity_curve"])

    _print_section("Test Metrics", test_metrics)
    _print_overfitting_check(train_metrics, test_metrics)

    # 5. Walk-forward (optional)
    wf_results = []
    if walk_fwd:
        print("\nWalk-forward validation…")
        wf_results = walk_forward(df, n_windows=5, train_days=20, test_days=10, n_trials=30)

    # 6. Re-run train for full trade log + charts
    train_prep_full  = add_all_indicators(train_df)
    train_result_full = run_backtest(train_prep_full, best_params)

    # 7. Save trade CSVs
    tag = f"{exchange}_{symbol}"
    train_result_full["trades"].to_csv(
        os.path.join(RESULTS_DIR, f"{tag}_train_trades.csv"), index=False)
    test_result["trades"].to_csv(
        os.path.join(RESULTS_DIR, f"{tag}_test_trades.csv"),  index=False)

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
    }
    with open(os.path.join(RESULTS_DIR, f"{tag}_summary.json"), "w") as f:
        # Don't write the base64 to JSON — too large
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
    """
    Build a 3-row dashboard.  Returns (file_path, base64_png).
    """
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
        ax3.hist(train_trades["pnl_pct"] * 100, bins=30, color=C_TRAIN, alpha=0.6, label="Train")
    if not test_trades.empty:
        ax3.hist(test_trades["pnl_pct"] * 100, bins=30, color=C_TEST, alpha=0.6, label="Test")
    ax3.axvline(0, color="#888", lw=1, ls="--")
    ax3.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax3, "Trade Return Distribution (%)")

    # 4 ── Exit reason breakdown
    ax4 = fig.add_subplot(gs[2, 0])
    for trades, color, lbl in [(train_trades, C_TRAIN, "Tr"), (test_trades, C_TEST, "Te")]:
        if not trades.empty:
            counts = trades["exit_reason"].value_counts()
            bars   = [f"{r}\n({lbl})" for r in counts.index]
            ax4.bar(bars, counts.values, color=color, alpha=0.75)
    ax4.tick_params(axis="x", labelsize=6)
    style(ax4, "Exit Reason Breakdown")

    # 5 ── Holding time
    ax5 = fig.add_subplot(gs[2, 1])
    if not train_trades.empty:
        ax5.hist(train_trades["holding_minutes"], bins=20, color=C_TRAIN, alpha=0.6, label="Train")
    if not test_trades.empty:
        ax5.hist(test_trades["holding_minutes"], bins=20, color=C_TEST, alpha=0.6, label="Test")
    ax5.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax5, "Holding Time (minutes)")

    fig.suptitle(
        f"VWAP Mean Reversion  |  {exchange}:{symbol}",
        color=TEXT, fontsize=12, fontweight="bold", y=0.99,
    )

    chart_path = os.path.join(RESULTS_DIR, f"{exchange}_{symbol}_charts.png")
    plt.savefig(chart_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    # Also encode as base64 for inline display
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()

    return chart_path, chart_b64


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_section(title: str, d: dict):
    print(f"\n── {title} ──")
    for k, v in d.items():
        print(f"  {k:<24s}: {v}")


def _print_overfitting_check(train: dict, test: dict):
    print("\n── Overfitting Check ──")
    t_sh, e_sh = train["sharpe_ratio"], test["sharpe_ratio"]
    t_rt, e_rt = train["total_return_pct"], test["total_return_pct"]
    ratio_sh = (e_sh / t_sh) if t_sh > 0 else 0
    ratio_rt = (e_rt / t_rt) if t_rt > 0 else 0
    print(f"  Sharpe  train={t_sh:.3f}  test={e_sh:.3f}  ratio={ratio_sh:.2f}")
    print(f"  Return  train={t_rt:.2f}%  test={e_rt:.2f}%  ratio={ratio_rt:.2f}")
    if ratio_sh < 0.5:
        print("  ⚠  Potential overfitting: test Sharpe < 50 % of train Sharpe")
    else:
        print("  ✓  Generalisation looks acceptable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VWAP Mean Reversion Backtester")
    parser.add_argument("--symbol",      default="INFY",  help="Stock symbol")
    parser.add_argument("--exchange",    default="NSE",   help="Exchange (NSE/BSE)")
    parser.add_argument("--optimize",    action="store_true", help="Run Bayesian optimisation")
    parser.add_argument("--trials",      type=int, default=100, help="Optimisation trials")
    parser.add_argument("--walkforward", action="store_true",   help="Run walk-forward validation")
    parser.add_argument("--all",         action="store_true",   help="Run for all configured stocks")
    args = parser.parse_args()

    if args.all:
        import json as _json
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_stocks.json")
        with open(cfg_path) as f:
            stocks = _json.load(f)["stocks"]
        for s in stocks:
            try:
                run_full_pipeline(
                    symbol          = s["symbol"],
                    exchange        = s["exchange"],
                    optimize_params = args.optimize,
                    n_trials        = args.trials,
                    walk_fwd        = args.walkforward,
                )
            except Exception as exc:
                print(f"  ERROR for {s['symbol']}: {exc}")
    else:
        run_full_pipeline(
            symbol          = args.symbol,
            exchange        = args.exchange,
            optimize_params = args.optimize,
            n_trials        = args.trials,
            walk_fwd        = args.walkforward,
        )

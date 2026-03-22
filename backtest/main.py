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

from backtest.data_loader         import load_csv, split_train_test
from backtest.indicators          import add_all_indicators
from backtest.strategy            import StrategyParams
from backtest.backtester          import run_backtest, run_backtest_ml
from backtest.metrics             import compute_metrics, aggregate_across_stocks
from backtest.optimizer           import optimize, optimize_cross_stock, walk_forward
from backtest.feature_engineering import add_ml_features, build_training_data
from backtest.regime              import add_regime, regime_summary
from backtest.ml_model            import TradeFilterModel
from backtest.position_sizing     import MLConfig, PositionSizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_ml(df: pd.DataFrame, ml_config: MLConfig) -> pd.DataFrame:
    """
    Full preparation pipeline for ML-enhanced backtesting:
      add_all_indicators → add_ml_features → add_regime
    """
    prep = add_all_indicators(df)
    prep = add_ml_features(prep)
    prep = add_regime(
        prep,
        vwap_slope_threshold  = ml_config.vwap_slope_threshold,
        regime_atr_multiplier = ml_config.regime_atr_multiplier,
    )
    return prep


def _train_ml_model(
    train_result: dict,
    train_prep:   pd.DataFrame,
    tag:          str = "UNIVERSE",
) -> TradeFilterModel | None:
    """
    Train a TradeFilterModel on trades from the training set.
    Returns None if there are not enough samples.
    """
    X, y = build_training_data(train_result["trades"], train_prep)
    if X.empty or len(y) < 10:
        print(f"  [ML] Not enough training samples ({len(y)}) — ML disabled.")
        return None

    pos_rate = float(y.mean()) * 100
    print(f"  [ML] Training on {len(y)} trades  (win rate: {pos_rate:.1f}%)")

    try:
        model = TradeFilterModel(n_estimators=200)
        model.fit(X, y)
        path = model.save(tag)
        print(f"  [ML] Model saved → {path}  (backend: {model._backend})")
        imp = model.feature_importance()
        top = list(imp.items())[:5]
        print(f"  [ML] Top features: {top}")
        return model
    except Exception as exc:
        print(f"  [ML] Training failed: {exc}")
        return None


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
    print(f"  Multi-Strategy Ensemble  |  {exchange}:{symbol}")
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
        f"Multi-Strategy Ensemble  |  {exchange}:{symbol}",
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
# Multi-stock pipeline  (NEW)
# ---------------------------------------------------------------------------

def run_all_pipeline(
    stocks_cfg:      list,
    optimize_params: bool  = False,
    n_trials:        int   = 100,
    default_params:  dict  = None,
) -> dict:
    """
    Run backtest across ALL stocks using a single universal StrategyParams,
    then train a per-stock ML model and run the ML-enhanced pass on the test set.

    Returns
    -------
    {
      "best_params":        dict,
      "per_stock":          [ {symbol, exchange, train_metrics, test_metrics,
                               ml_test_metrics, chart_b64, ml_chart_b64,
                               feature_importance, regime_summary}, … ],
      "aggregate":          {portfolio_metrics, avg_metrics, …},
      "ml_aggregate":       same structure but for ML test results,
      "combined_chart_b64": str,
      "ml_combined_chart_b64": str,
    }
    """
    print(f"\n{'='*60}")
    print(f"  Multi-Strategy Ensemble + ML  |  {len(stocks_cfg)} stocks")
    print(f"{'='*60}")

    ml_config = MLConfig()
    sizer     = PositionSizer()

    # 1. Load and split every stock
    loaded = []
    for s in stocks_cfg:
        try:
            df = load_csv(s["symbol"], s["exchange"])
            train_df, test_df, train_days, test_days = split_train_test(df)
            loaded.append({
                "symbol":    s["symbol"],
                "exchange":  s["exchange"],
                "train_df":  train_df,
                "test_df":   test_df,
                "train_days": train_days,
                "test_days":  test_days,
            })
            print(f"  Loaded {s['symbol']}: {len(df):,} candles")
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

    # 2. Determine strategy params (same for base and ML runs)
    if optimize_params:
        print(f"\nCross-stock Bayesian optimisation ({n_trials} trials)…")
        best_params, avg_score = optimize_cross_stock(
            [s["train_df"] for s in loaded],
            n_trials=n_trials,
            show_progress=True,
        )
        print(f"  avg train score: {avg_score:.4f}")
    else:
        best_params = StrategyParams.from_dict(default_params or {})

    _print_section("Universal Parameters", best_params.to_dict())

    # 3. Evaluate every stock — base + ML
    per_stock_results = []
    for s in loaded:
        sym, exch = s["symbol"], s["exchange"]
        tag = f"{exch}_{sym}"
        try:
            # ── Base run on train (also used to generate ML training data) ──
            train_prep    = _prepare_ml(s["train_df"], ml_config)
            train_result  = run_backtest(train_prep, best_params)
            train_metrics = compute_metrics(train_result["trades"],
                                            train_result["equity_curve"])

            # ── Base run on test ─────────────────────────────────────────────
            test_prep    = _prepare_ml(s["test_df"], ml_config)
            test_result  = run_backtest(test_prep, best_params)
            test_metrics = compute_metrics(test_result["trades"],
                                           test_result["equity_curve"])

            # ── Train ML model on training trades ────────────────────────────
            print(f"\n  [{sym}] Training ML model…")
            reg_info = regime_summary(train_prep)
            print(f"  [{sym}] Regime (train): {reg_info}")
            model = _train_ml_model(train_result, train_prep, tag=tag)

            # ── ML-enhanced run on test ──────────────────────────────────────
            ml_test_result  = run_backtest_ml(test_prep, best_params,
                                              model=model,
                                              ml_config=ml_config,
                                              sizer=sizer)
            ml_test_metrics = compute_metrics(ml_test_result["trades"],
                                              ml_test_result["equity_curve"])

            feat_imp = model.feature_importance() if model else {}

            print(
                f"  {sym:<12s}"
                f"  base Sharpe={test_metrics['sharpe_ratio']:.2f}"
                f"  ML Sharpe={ml_test_metrics['sharpe_ratio']:.2f}"
                f"  trades {test_metrics['n_trades']}→{ml_test_metrics['n_trades']}"
            )

            # ── Charts ───────────────────────────────────────────────────────
            _, chart_b64 = generate_charts(
                train_result["equity_curve"], test_result["equity_curve"],
                train_result["trades"],       test_result["trades"],
                sym, exch,
            )
            _, ml_chart_b64 = generate_ml_charts(
                test_result,
                ml_test_result,
                feat_imp,
                sym, exch,
            )

            per_stock_results.append({
                "symbol":          sym,
                "exchange":        exch,
                "train_metrics":   train_metrics,
                "test_metrics":    test_metrics,
                "ml_test_metrics": ml_test_metrics,
                "chart_b64":       chart_b64,
                "ml_chart_b64":    ml_chart_b64,
                "feature_importance": feat_imp,
                "regime_summary":  reg_info,
                # kept for aggregation
                "test_trades":     test_result["trades"],
                "test_equity":     test_result["equity_curve"],
                "ml_test_trades":  ml_test_result["trades"],
                "ml_test_equity":  ml_test_result["equity_curve"],
            })

        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  ERROR {sym}: {exc}")
            empty = pd.DataFrame()
            empty_s = pd.Series(dtype=float)
            per_stock_results.append({
                "symbol":          sym,
                "exchange":        exch,
                "train_metrics":   {},
                "test_metrics":    {},
                "ml_test_metrics": {},
                "chart_b64":       "",
                "ml_chart_b64":    "",
                "feature_importance": {},
                "regime_summary":  {},
                "test_trades":     empty,
                "test_equity":     empty_s,
                "ml_test_trades":  empty,
                "ml_test_equity":  empty_s,
                "error":           str(exc),
            })

    # 4. Aggregate — base
    aggregate    = aggregate_across_stocks(per_stock_results)

    # 4b. Aggregate — ML (swap in ml trades/equity)
    ml_proxy = []
    for r in per_stock_results:
        ml_proxy.append({
            **r,
            "test_trades":  r.get("ml_test_trades", pd.DataFrame()),
            "test_equity":  r.get("ml_test_equity", pd.Series(dtype=float)),
            "test_metrics": r.get("ml_test_metrics", {}),
        })
    ml_aggregate = aggregate_across_stocks(ml_proxy)

    # 5. Combined charts
    combined_b64    = generate_combined_chart(per_stock_results)
    ml_combined_b64 = generate_combined_chart(
        ml_proxy,
        title="ML-Enhanced Cross-Stock Results  |  Universal + ML Filter",
    )

    # 6. Save params JSON
    with open(os.path.join(RESULTS_DIR, "UNIVERSE_params.json"), "w") as f:
        json.dump(best_params.to_dict(), f, indent=2)
    with open(os.path.join(RESULTS_DIR, "UNIVERSE_ml_config.json"), "w") as f:
        json.dump(ml_config.to_dict(), f, indent=2)

    # Strip DataFrames before returning
    for r in per_stock_results:
        r.pop("test_trades",    None)
        r.pop("test_equity",    None)
        r.pop("ml_test_trades", None)
        r.pop("ml_test_equity", None)

    return {
        "best_params":           best_params.to_dict(),
        "ml_config":             ml_config.to_dict(),
        "per_stock":             per_stock_results,
        "aggregate":             aggregate,
        "ml_aggregate":          ml_aggregate,
        "combined_chart_b64":    combined_b64,
        "ml_combined_chart_b64": ml_combined_b64,
        "n_configured":          n_configured,
        "n_loaded":              n_loaded,
    }


def generate_ml_charts(
    base_result:  dict,
    ml_result:    dict,
    feat_imp:     dict,
    symbol:       str,
    exchange:     str,
) -> tuple[str, str]:
    """
    4-panel ML comparison chart:
      Row 1: Base vs ML test equity curves (overlay)
      Row 2: ML predicted probability distribution
      Row 3: Feature importance bar chart
      Row 4: Base vs ML per-metric comparison table (bar)
    Returns (file_path, base64_png).
    """
    C_BASE  = "#dc2626"
    C_ML    = "#22c55e"
    BG      = "#181818"
    GRID    = "#2a2a2a"
    TEXT    = "#cccccc"

    fig = plt.figure(figsize=(14, 11), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors=TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    base_eq = base_result["equity_curve"]
    ml_eq   = ml_result["equity_curve"]
    ml_trades = ml_result["trades"]

    # ── Panel 1: equity curve comparison ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(len(base_eq)), base_eq.values,
             color=C_BASE, lw=1.4, label=f"Base ({base_result['n_trades']} trades)")
    ax1.plot(range(len(ml_eq)), ml_eq.values,
             color=C_ML,   lw=1.4, label=f"ML   ({ml_result['n_trades']} trades)")
    ax1.axhline(0, color="#555", lw=0.8, ls="--")
    ax1.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax1, f"Base vs ML-Enhanced  |  Test Equity  |  {exchange}:{symbol}")

    # ── Panel 2: ML probability distribution ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if not ml_trades.empty and "ml_prob" in ml_trades.columns:
        probs = ml_trades["ml_prob"].dropna()
        wins  = ml_trades.loc[ml_trades["pnl_pct"] > 0, "ml_prob"].dropna()
        loss  = ml_trades.loc[ml_trades["pnl_pct"] <= 0, "ml_prob"].dropna()
        bins  = np.linspace(0, 1, 21)
        ax2.hist(wins.values, bins=bins, color=C_ML,   alpha=0.6, label="Winners")
        ax2.hist(loss.values, bins=bins, color=C_BASE, alpha=0.6, label="Losers")
        ax2.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    else:
        ax2.text(0.5, 0.5, "No ML trades", ha="center", va="center",
                 color=TEXT, transform=ax2.transAxes)
    style(ax2, "ML Predicted Probability Distribution")

    # ── Panel 3: feature importance ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if feat_imp:
        names = list(feat_imp.keys())[:8]
        vals  = [feat_imp[n] for n in names]
        colors_bar = [C_ML if v == max(vals) else "#3b82f6" for v in vals]
        bars = ax3.barh(names[::-1], vals[::-1], color=colors_bar[::-1], alpha=0.85)
        for bar, val in zip(bars, vals[::-1]):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=6.5, color=TEXT)
        ax3.set_xlabel("Importance", color=TEXT, fontsize=7)
    else:
        ax3.text(0.5, 0.5, "No model trained", ha="center", va="center",
                 color=TEXT, transform=ax3.transAxes)
    style(ax3, "Feature Importance")

    fig.suptitle(
        f"ML Analysis  |  {exchange}:{symbol}",
        color=TEXT, fontsize=12, fontweight="bold", y=0.99,
    )

    tag = f"{exchange}_{symbol}"
    chart_path = os.path.join(RESULTS_DIR, f"{tag}_ml_charts.png")
    plt.savefig(chart_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    return chart_path, chart_b64


def generate_combined_chart(per_stock_results: list, title: str = "") -> str:
    """
    Combined chart showing:
      Row 1: Portfolio cumulative equity (sum of all test equity curves)
      Row 2: Per-stock test equity curves (normalised to start at 0)
      Row 3: Bar chart of per-stock test Sharpe ratios
    Returns base64-encoded PNG.
    """
    C_PORTFOLIO = "#dc2626"
    BG    = "#181818"
    GRID  = "#2a2a2a"
    TEXT  = "#cccccc"

    # Colour palette for individual stocks
    PALETTE = [
        "#22c55e", "#3b82f6", "#f59e0b", "#a855f7",
        "#06b6d4", "#f97316", "#ec4899", "#84cc16",
    ]

    fig = plt.figure(figsize=(14, 11), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors=TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # Gather normalised test equity curves
    eq_list    = []
    labels     = []
    for r in per_stock_results:
        eq = r.get("test_equity")
        if eq is not None and not eq.empty:
            eq_list.append(eq.reset_index(drop=True))
            labels.append(r["symbol"])

    # ── Row 1: Portfolio equity ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    if eq_list:
        min_len      = min(len(e) for e in eq_list)
        portfolio_eq = sum(e.iloc[:min_len] for e in eq_list)
        ax1.plot(portfolio_eq.values, color=C_PORTFOLIO, lw=1.8,
                 label=f"Portfolio ({len(eq_list)} stocks)")
        ax1.axhline(0, color="#555", lw=0.8, ls="--")
        ax1.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    style(ax1, "Portfolio Test Equity (equal-weight sum of all stocks)")

    # ── Row 2: Per-stock equity ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for i, (eq, lbl) in enumerate(zip(eq_list, labels)):
        color = PALETTE[i % len(PALETTE)]
        ax2.plot(eq.values, color=color, lw=0.9, alpha=0.85, label=lbl)
    if eq_list:
        ax2.axhline(0, color="#555", lw=0.8, ls="--")
        ax2.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
                   ncol=min(len(labels), 6))
    style(ax2, "Per-Stock Test Equity (normalised)")

    # ── Row 3: Sharpe bar chart ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    syms   = [r["symbol"] for r in per_stock_results]
    sharpes = [r.get("test_metrics", {}).get("sharpe_ratio", 0.0) for r in per_stock_results]
    colors  = [PALETTE[i % len(PALETTE)] if s > 0 else "#6b7280"
               for i, s in enumerate(sharpes)]
    bars = ax3.bar(syms, sharpes, color=colors, alpha=0.85)
    ax3.axhline(0, color="#888", lw=0.8)
    ax3.tick_params(axis="x", labelsize=7, rotation=30)
    # Annotate bars
    for bar, val in zip(bars, sharpes):
        if val != 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, val,
                     f"{val:.2f}", ha="center",
                     va="bottom" if val > 0 else "top",
                     fontsize=6.5, color=TEXT)
    style(ax3, "Per-Stock Test Sharpe Ratio")

    fig.suptitle(
        title or "Cross-Stock Backtest Results  |  Universal Parameters",
        color=TEXT, fontsize=12, fontweight="bold", y=0.99,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.savefig(
        os.path.join(RESULTS_DIR, "UNIVERSE_combined_chart.png"),
        dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    return b64


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

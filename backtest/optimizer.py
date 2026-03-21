"""
Parameter optimisation for the VWAP Mean-Reversion strategy.

Primary:  Bayesian optimisation via Optuna (TPE sampler).
Fallback: Random search (if optuna is not installed).

Also provides walk-forward validation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from .strategy import StrategyParams, PARAM_BOUNDS
from .backtester import run_backtest
from .metrics import compute_metrics, objective_score
from .indicators import add_all_indicators


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def optimize(
    train_df,
    n_trials: int = 100,
    show_progress: bool = False,
) -> Tuple[StrategyParams, Dict]:
    """
    Optimise strategy parameters on `train_df`.

    Uses Optuna (Bayesian / TPE) when available, otherwise random search.
    Returns (best_params, train_metrics).
    """
    try:
        import optuna  # noqa: F401
        return _optuna_optimize(train_df, n_trials, show_progress)
    except ImportError:
        return _random_search(train_df, n_trials)


# ---------------------------------------------------------------------------
# Optuna (Bayesian)
# ---------------------------------------------------------------------------

def _optuna_optimize(
    train_df,
    n_trials: int,
    show_progress: bool,
) -> Tuple[StrategyParams, Dict]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute indicators once — the objective reuses this prepared frame
    prepared = add_all_indicators(train_df)

    def objective(trial):
        params = StrategyParams(
            threshold         = trial.suggest_float("threshold",         *PARAM_BOUNDS["threshold"]),
            volume_filter     = trial.suggest_float("volume_filter",     *PARAM_BOUNDS["volume_filter"]),
            stop_loss         = trial.suggest_float("stop_loss",         *PARAM_BOUNDS["stop_loss"]),
            take_profit       = trial.suggest_float("take_profit",       *PARAM_BOUNDS["take_profit"]),
            max_holding       = trial.suggest_int(  "max_holding",       *PARAM_BOUNDS["max_holding"]),
            time_open_filter  = trial.suggest_int(  "time_open_filter",  *PARAM_BOUNDS["time_open_filter"]),
            time_close_filter = trial.suggest_int(  "time_close_filter", *PARAM_BOUNDS["time_close_filter"]),
        )
        result  = run_backtest(prepared, params)
        metrics = compute_metrics(result["trades"], result["equity_curve"])
        return objective_score(metrics)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best = StrategyParams(**{k: study.best_params[k] for k in PARAM_BOUNDS})
    result  = run_backtest(prepared, best)
    metrics = compute_metrics(result["trades"], result["equity_curve"])
    return best, metrics


# ---------------------------------------------------------------------------
# Random search fallback
# ---------------------------------------------------------------------------

def _random_search(train_df, n_trials: int) -> Tuple[StrategyParams, Dict]:
    rng      = np.random.default_rng(42)
    prepared = add_all_indicators(train_df)

    best_score   = -np.inf
    best_params  = StrategyParams()
    best_metrics = {}

    for _ in range(n_trials):
        params = StrategyParams(
            threshold         = float(rng.uniform(*PARAM_BOUNDS["threshold"])),
            volume_filter     = float(rng.uniform(*PARAM_BOUNDS["volume_filter"])),
            stop_loss         = float(rng.uniform(*PARAM_BOUNDS["stop_loss"])),
            take_profit       = float(rng.uniform(*PARAM_BOUNDS["take_profit"])),
            max_holding       = int(rng.integers(*PARAM_BOUNDS["max_holding"])),
            time_open_filter  = int(rng.integers(*PARAM_BOUNDS["time_open_filter"])),
            time_close_filter = int(rng.integers(*PARAM_BOUNDS["time_close_filter"])),
        )
        result  = run_backtest(prepared, params)
        metrics = compute_metrics(result["trades"], result["equity_curve"])
        score   = objective_score(metrics)

        if score > best_score:
            best_score   = score
            best_params  = params
            best_metrics = metrics

    return best_params, best_metrics


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward(
    df,
    n_windows:   int = 5,
    train_days:  int = 30,
    test_days:   int = 10,
    n_trials:    int = 50,
) -> List[Dict]:
    """
    Rolling walk-forward optimisation.

    Each window:
      - optimise on train_days of data
      - evaluate best params on the immediately following test_days

    Returns a list of result dicts, one per window.
    """
    unique_days = sorted(df["date"].unique())
    results     = []

    for w in range(n_windows):
        start     = w * test_days
        train_end = start + train_days
        test_end  = train_end + test_days

        if test_end > len(unique_days):
            break

        w_train_days = unique_days[start:train_end]
        w_test_days  = unique_days[train_end:test_end]

        w_train = df[df["date"].isin(w_train_days)].reset_index(drop=True)
        w_test  = df[df["date"].isin(w_test_days)].reset_index(drop=True)

        best_params, train_metrics = optimize(w_train, n_trials=n_trials, show_progress=False)

        test_prep    = add_all_indicators(w_test)
        test_result  = run_backtest(test_prep, best_params)
        test_metrics = compute_metrics(test_result["trades"], test_result["equity_curve"])

        results.append({
            "window":       w + 1,
            "train_period": f"{w_train_days[0]} → {w_train_days[-1]}",
            "test_period":  f"{w_test_days[0]}  → {w_test_days[-1]}",
            "train_sharpe": train_metrics["sharpe_ratio"],
            "test_sharpe":  test_metrics["sharpe_ratio"],
            "train_return": train_metrics["total_return_pct"],
            "test_return":  test_metrics["total_return_pct"],
            "train_trades": train_metrics["n_trades"],
            "test_trades":  test_metrics["n_trades"],
            "best_params":  best_params.to_dict(),
        })

        print(
            f"  WF Window {w+1}: "
            f"Train Sharpe={train_metrics['sharpe_ratio']:.2f}  "
            f"Test Sharpe={test_metrics['sharpe_ratio']:.2f}"
        )

    return results

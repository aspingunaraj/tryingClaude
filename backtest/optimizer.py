"""
Parameter optimisation for the multi-strategy ensemble system.

Primary:  Bayesian optimisation via Optuna (TPE sampler).
Fallback: Random search (if optuna is not installed).

All 9 parameters in PARAM_BOUNDS are optimised.  INT_PARAMS controls which
are sampled as integers.  Cost params (slippage, commission) are fixed.

Objective function penalises:
  - high max drawdown  (×2 weight)
  - too few trades     (< 10 trades)

Cross-stock mode: one universal StrategyParams is found that maximises the
average objective score across ALL supplied training DataFrames.

Walk-forward validation: rolling train/test windows.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from .strategy  import StrategyParams, PARAM_BOUNDS, INT_PARAMS
from .backtester import run_backtest
from .metrics   import compute_metrics, objective_score
from .indicators import add_all_indicators


# ---------------------------------------------------------------------------
# Helpers: build a StrategyParams from a trial (Optuna) or RNG (random search)
# ---------------------------------------------------------------------------

def _params_from_trial(trial) -> StrategyParams:
    """Sample every PARAM_BOUNDS key from an Optuna trial."""
    kwargs = {}
    for k, (lo, hi) in PARAM_BOUNDS.items():
        if k in INT_PARAMS:
            kwargs[k] = trial.suggest_int(k, lo, hi)
        else:
            kwargs[k] = trial.suggest_float(k, lo, hi)
    return StrategyParams(**kwargs)


def _params_from_rng(rng) -> StrategyParams:
    """Sample every PARAM_BOUNDS key from a numpy RNG (random search fallback)."""
    kwargs = {}
    for k, (lo, hi) in PARAM_BOUNDS.items():
        if k in INT_PARAMS:
            kwargs[k] = int(rng.integers(lo, hi + 1))
        else:
            kwargs[k] = float(rng.uniform(lo, hi))
    return StrategyParams(**kwargs)


def _best_params_from_study(study) -> StrategyParams:
    """Reconstruct the best StrategyParams from a completed Optuna study."""
    return StrategyParams(**{k: study.best_params[k] for k in PARAM_BOUNDS})


# ---------------------------------------------------------------------------
# Single-stock optimisation
# ---------------------------------------------------------------------------

def optimize(
    train_df,
    n_trials:      int  = 200,
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
# Cross-stock optimisation
# ---------------------------------------------------------------------------

def optimize_cross_stock(
    train_dfs:     List,
    n_trials:      int  = 200,
    show_progress: bool = False,
) -> Tuple[StrategyParams, float]:
    """
    Find a single StrategyParams that maximises the *average* objective score
    across every stock in `train_dfs` (list of raw train DataFrames).

    Indicators are pre-computed once per stock before the search begins so
    each trial is fast.  Returns (best_params, avg_train_score).
    """
    prepared_list = [
        add_all_indicators(df)
        for df in train_dfs
        if df is not None and len(df) > 0
    ]
    if not prepared_list:
        return StrategyParams(), 0.0

    try:
        import optuna  # noqa: F401
        return _optuna_cross_stock(prepared_list, n_trials, show_progress)
    except ImportError:
        return _random_search_cross_stock(prepared_list, n_trials)


def _cross_stock_score(prepared_list: List, params: StrategyParams) -> float:
    """Average objective score across all stocks; skips stocks with errors."""
    scores = []
    for prep in prepared_list:
        try:
            result  = run_backtest(prep, params)
            metrics = compute_metrics(result["trades"], result["equity_curve"])
            s = objective_score(metrics)
            if s > -100:
                scores.append(s)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else -999.0


# ---------------------------------------------------------------------------
# Optuna cross-stock
# ---------------------------------------------------------------------------

def _optuna_cross_stock(
    prepared_list: List,
    n_trials:      int,
    show_progress: bool,
) -> Tuple[StrategyParams, float]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        return _cross_stock_score(prepared_list, _params_from_trial(trial))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best  = _best_params_from_study(study)
    score = _cross_stock_score(prepared_list, best)
    return best, score


# ---------------------------------------------------------------------------
# Random search cross-stock fallback
# ---------------------------------------------------------------------------

def _random_search_cross_stock(
    prepared_list: List,
    n_trials:      int,
) -> Tuple[StrategyParams, float]:
    rng         = np.random.default_rng(42)
    best_score  = -np.inf
    best_params = StrategyParams()

    for _ in range(n_trials):
        params = _params_from_rng(rng)
        score  = _cross_stock_score(prepared_list, params)
        if score > best_score:
            best_score  = score
            best_params = params

    return best_params, best_score


# ---------------------------------------------------------------------------
# Optuna single-stock (Bayesian)
# ---------------------------------------------------------------------------

def _optuna_optimize(
    train_df,
    n_trials:      int,
    show_progress: bool,
) -> Tuple[StrategyParams, Dict]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    prepared = add_all_indicators(train_df)

    def objective(trial):
        params  = _params_from_trial(trial)
        result  = run_backtest(prepared, params)
        metrics = compute_metrics(result["trades"], result["equity_curve"])
        return objective_score(metrics)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best    = _best_params_from_study(study)
    result  = run_backtest(prepared, best)
    metrics = compute_metrics(result["trades"], result["equity_curve"])
    return best, metrics


# ---------------------------------------------------------------------------
# Random search fallback (single-stock)
# ---------------------------------------------------------------------------

def _random_search(train_df, n_trials: int) -> Tuple[StrategyParams, Dict]:
    rng      = np.random.default_rng(42)
    prepared = add_all_indicators(train_df)

    best_score   = -np.inf
    best_params  = StrategyParams()
    best_metrics: Dict = {}

    for _ in range(n_trials):
        params  = _params_from_rng(rng)
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
    n_windows:  int = 5,
    train_days: int = 30,
    test_days:  int = 10,
    n_trials:   int = 50,
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

        best_params, train_metrics = optimize(w_train, n_trials=n_trials,
                                              show_progress=False)

        test_prep    = add_all_indicators(w_test)
        test_result  = run_backtest(test_prep, best_params)
        test_metrics = compute_metrics(test_result["trades"], test_result["equity_curve"])

        results.append({
            "window":       w + 1,
            "train_period": f"{w_train_days[0]} → {w_train_days[-1]}",
            "test_period":  f"{w_test_days[0]}  → {w_test_days[-1]}",
            "train_sharpe": train_metrics.get("sharpe_ratio", 0.0),
            "test_sharpe":  test_metrics.get("sharpe_ratio", 0.0),
            "train_return": train_metrics.get("total_return_pct", 0.0),
            "test_return":  test_metrics.get("total_return_pct", 0.0),
            "train_trades": train_metrics.get("n_trades", 0),
            "test_trades":  test_metrics.get("n_trades", 0),
            "best_params":  best_params.to_dict(),
        })

        print(
            f"  WF Window {w+1}: "
            f"Train Sharpe={train_metrics.get('sharpe_ratio', 0):.2f}  "
            f"Test Sharpe={test_metrics.get('sharpe_ratio', 0):.2f}"
        )

    return results

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # scipy is a runtime dep (requirements.in); guard so import never hard-fails
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:  # pragma: no cover - exercised only on a stripped install
    _scipy_stats = None
    _SCIPY = False


def run_one_episode(model, vec_env, deterministic: bool = True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        # MaskablePPO needs the current action mask at predict time; pull it
        # from the env(s) and pass it through (review #1: _predict was undefined).
        action_masks = np.asarray(vec_env.env_method("action_masks"))
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity = info.get("equity_usd", vec_env.get_attr("equity_usd")[0])
        equity_curve.append(float(equity))

        trade_info = info.get("last_trade_info")
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    final_equity = float(equity_curve[-1]) if equity_curve else float(vec_env.get_attr("equity_usd")[0])
    return equity_curve, final_equity, closed_trades


def compute_metrics(closed_trades: list, equity_curve: list, initial_equity: float = 10000.0,
                    bar_hours: float = 1.0):
    if not closed_trades or not equity_curve:
        return {
            "sharpe": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade_pips": 0.0,
            "n_trades": 0,
            "max_dd": 0.0,
        }

    net_pips = [trade["net_pips"] for trade in closed_trades]
    wins = [pips for pips in net_pips if pips > 0]
    losses = [pips for pips in net_pips if pips <= 0]

    win_rate = len(wins) / len(net_pips)
    avg_trade_pips = float(np.mean(net_pips))
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_wins / gross_losses

    equity_arr = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (peak - equity_arr) / np.maximum(peak, 1e-9)
    max_dd = float(np.max(drawdowns))

    returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-9)
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        # Annualization factor must reflect the actual bar size (review #5):
        # bars per year = 252 trading days * 24h / bar_hours.
        bars_per_year = 252 * 24 / max(bar_hours, 1e-9)
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year))

    total_return = (equity_arr[-1] - initial_equity) / initial_equity
    calmar = float(total_return / max_dd) if max_dd > 0 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 3),
        "avg_trade_pips": round(avg_trade_pips, 3),
        "n_trades": len(net_pips),
        "max_dd": round(max_dd, 3),
    }


def score_model(model, vec_env, bar_hours: float = 1.0, deterministic: bool = True):
    """Run one evaluation episode and return a stable selection score.

    Final equity alone is path-dependent (review #10); Sharpe is a steadier
    signal for picking the best checkpoint. Final equity is returned too as a
    tiebreak / for logging.
    """
    equity_curve, final_equity, closed_trades = run_one_episode(model, vec_env, deterministic)
    metrics = compute_metrics(closed_trades, equity_curve, bar_hours=bar_hours)
    return metrics["sharpe"], final_equity, equity_curve, closed_trades


def save_trade_history(closed_trades: list[dict], output_path: str | Path) -> Path | None:
    if not closed_trades:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(closed_trades).to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Phase 3.6 — statistical testing on results
# ---------------------------------------------------------------------------
#
# A point estimate of Sharpe or final equity is not a result you can defend; it
# is one draw from a distribution. These helpers turn a single backtest episode
# into claims with uncertainty attached:
#   * a one-sample t-test that the mean per-bar return is actually > 0,
#   * a bootstrap 95% CI for the per-trade Sharpe,
#   * the Deflated Sharpe Ratio (Lopez de Prado, 2014), which discounts an
#     observed Sharpe for the number of configurations that were tried (e.g.
#     Optuna trials) before it was selected.
#
# Everything here is SB3-free on purpose so the stats can be unit-tested without
# the heavy RL stack installed.


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF. Uses scipy when present, else math.erf."""
    if _SCIPY:
        return float(_scipy_stats.norm.cdf(x))
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile)."""
    if _SCIPY:
        return float(_scipy_stats.norm.ppf(p))
    # Acklam's rational approximation — adequate for the N-trial expected-max term.
    import math
    if not 0.0 < p < 1.0:
        raise ValueError(f"ppf argument must be in (0, 1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def returns_ttest(equity_curve: list, alternative: str = "greater") -> dict:
    """One-sample t-test on per-bar simple returns: is the mean return > 0?

    Returns t-statistic, (one-sided by default) p-value, sample size and mean.
    With < 2 returns or zero variance the test is undefined and p is reported as
    ``None`` rather than a misleading number.
    """
    equity_arr = np.asarray(equity_curve, dtype=np.float64)
    if equity_arr.size < 3:
        return {"t_stat": None, "p_value": None, "n": int(max(equity_arr.size - 1, 0)),
                "mean_return": 0.0, "alternative": alternative}

    returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-9)
    n = int(returns.size)
    mean_ret = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if n < 2 or std == 0.0:
        return {"t_stat": None, "p_value": None, "n": n, "mean_return": round(mean_ret, 8),
                "alternative": alternative}

    if _SCIPY:
        res = _scipy_stats.ttest_1samp(returns, popmean=0.0, alternative=alternative)
        t_stat, p_value = float(res.statistic), float(res.pvalue)
    else:  # manual t-stat + normal-approx p-value fallback
        t_stat = mean_ret / (std / np.sqrt(n))
        if alternative == "greater":
            p_value = 1.0 - _norm_cdf(t_stat)
        elif alternative == "less":
            p_value = _norm_cdf(t_stat)
        else:
            p_value = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))

    return {"t_stat": round(t_stat, 4), "p_value": round(p_value, 6), "n": n,
            "mean_return": round(mean_ret, 8), "alternative": alternative}


def _sharpe_of(sample: np.ndarray) -> float:
    """Sharpe of a 1-D sample (mean / std, ddof=1). 0.0 when undefined."""
    if sample.size < 2:
        return 0.0
    std = np.std(sample, ddof=1)
    if std == 0.0:
        return 0.0
    return float(np.mean(sample) / std)


def bootstrap_sharpe_ci(per_trade_values: list, n_resamples: int = 10_000,
                        confidence: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap CI for the per-trade Sharpe ratio.

    Resamples the per-trade series (R-multiples are the recommended input — see
    ``r_multiples_from_trades``) with replacement ``n_resamples`` times and reports
    the point estimate plus the [(1-c)/2, 1-(1-c)/2] percentile interval. This is
    the Sharpe of the *trade* distribution, not annualized; it answers "is the
    edge per trade distinguishable from zero?".
    """
    values = np.asarray(per_trade_values, dtype=np.float64)
    n = values.size
    if n < 3:
        return {"sharpe": _sharpe_of(values), "ci_low": None, "ci_high": None,
                "n_trades": int(n), "n_resamples": int(n_resamples),
                "confidence": confidence}

    point = _sharpe_of(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    resampled = values[idx]
    means = resampled.mean(axis=1)
    stds = resampled.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(stds > 0, means / stds, 0.0)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(sharpes, 100 * alpha))
    ci_high = float(np.percentile(sharpes, 100 * (1.0 - alpha)))
    return {"sharpe": round(point, 4), "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4), "n_trades": int(n),
            "n_resamples": int(n_resamples), "confidence": confidence}


def r_multiples_from_trades(closed_trades: list) -> list:
    """Pull per-trade R-multiples (PnL in units of initial risk) from a trade log.

    Falls back to net_pips when the r_multiple key is absent (older trade logs).
    """
    out = []
    for trade in closed_trades:
        if "r_multiple" in trade and trade["r_multiple"] is not None:
            out.append(float(trade["r_multiple"]))
        elif "net_pips" in trade:
            out.append(float(trade["net_pips"]))
    return out


def deflated_sharpe_ratio(per_trade_values: list, n_trials: int = 1,
                          sr_benchmark: float = 0.0,
                          trial_sharpe_std: float | None = None) -> dict:
    """Deflated Sharpe Ratio (Lopez de Prado, 2014).

    Corrects an observed (non-annualized, per-trade) Sharpe for:
      * selection bias from testing ``n_trials`` configurations — the expected
        maximum Sharpe under the null grows with the number of trials,
      * non-normal returns (skew and excess kurtosis inflate Sharpe variance).

    ``trial_sharpe_std`` is the cross-trial dispersion of Sharpe estimates; when
    not supplied (single run, no sweep) it defaults to the analytic SE of the
    Sharpe estimator, ``sqrt((1 - skew*SR + (k-1)/4 * SR^2) / (T-1))``.

    Returns the DSR (a probability in [0, 1]) and the deflated benchmark SR0. A
    DSR > 0.95 is the usual "survives multiple testing" bar.
    """
    values = np.asarray(per_trade_values, dtype=np.float64)
    T = values.size
    if T < 3:
        return {"dsr": None, "observed_sharpe": _sharpe_of(values), "sr0": None,
                "n_trials": int(n_trials), "T": int(T)}

    sr = _sharpe_of(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0.0:
        return {"dsr": None, "observed_sharpe": 0.0, "sr0": None,
                "n_trials": int(n_trials), "T": int(T)}

    z = (values - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # non-excess (normal == 3)

    # Variance of the Sharpe estimator under non-normality (Mertens / Lo).
    sharpe_var = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2) / (T - 1)
    sharpe_var = max(sharpe_var, 1e-12)
    sharpe_se = np.sqrt(sharpe_var)

    # Expected maximum Sharpe under the null across n_trials independent trials.
    sr_std = float(trial_sharpe_std) if trial_sharpe_std is not None else float(sharpe_se)
    n = max(int(n_trials), 1)
    if n > 1 and sr_std > 0:
        euler = 0.5772156649015329
        e = np.e
        expected_max = sr_std * (
            (1.0 - euler) * _norm_ppf(1.0 - 1.0 / n)
            + euler * _norm_ppf(1.0 - 1.0 / (n * e))
        )
    else:
        expected_max = 0.0
    sr0 = sr_benchmark + expected_max

    dsr = _norm_cdf((sr - sr0) / sharpe_se)
    return {"dsr": round(float(dsr), 4), "observed_sharpe": round(sr, 4),
            "sr0": round(float(sr0), 4), "sharpe_se": round(float(sharpe_se), 4),
            "skew": round(skew, 4), "kurtosis": round(kurt, 4),
            "n_trials": int(n), "T": int(T)}


def statistical_report(equity_curve: list, closed_trades: list, n_trials: int = 1,
                       n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Bundle the Phase 3.6 tests into one report dict.

    ``n_trials`` should be the number of configurations evaluated before this one
    was chosen (e.g. Optuna trials) so the Deflated Sharpe penalizes selection.
    """
    r_multiples = r_multiples_from_trades(closed_trades)
    return {
        "returns_ttest": returns_ttest(equity_curve),
        "bootstrap_sharpe_ci": bootstrap_sharpe_ci(r_multiples, n_resamples=n_resamples, seed=seed),
        "deflated_sharpe": deflated_sharpe_ratio(r_multiples, n_trials=n_trials),
        "n_trials": int(n_trials),
    }

"""Shared helpers for evaluating a saved model on an arbitrary date slice.

Phase 3.7 (holdout) and 3.8 (regime-stratified) both need the same thing:
take a trained MaskablePPO model, feed it a slice of fully-featured data, run a
single deterministic episode through the exact training-time env + observation
normalization, and report metrics plus the Phase 3.6 statistical battery.

Keeping that logic here means the holdout and regime evaluators stay thin and
can never drift apart in how they build the env or normalize observations —
which would silently invalidate any comparison between them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    assert_feature_schema,
    build_bot_config,
    load_feature_columns,
    load_run_config,
    model_config_path_for,
)
from env_factory import make_eval_env
from evaluation import compute_metrics, run_one_episode, statistical_report


def slice_by_date(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Return rows of a UTC-DatetimeIndexed frame in [start, end).

    ``start``/``end`` are parsed as UTC. ``start`` is inclusive, ``end`` is
    exclusive (so "2024-01-01".."2024-07-01" is exactly H1 2024). Either bound
    may be None. The input must keep its DatetimeIndex (do not call this on a
    frame the env has already reset_index'd).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("slice_by_date requires a DatetimeIndex (call before the env reset_index).")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df.index >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        mask &= df.index < pd.Timestamp(end, tz="UTC")
    return df.loc[mask].copy()


def load_model_and_config(model_path: str | Path):
    """Load a saved MaskablePPO model and its run-config sidecar.

    Returns (model, config, config_path). Falls back to default config when no
    sidecar is present (older models).
    """
    from sb3_contrib import MaskablePPO  # local import keeps this module light to import

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    config_path = model_config_path_for(model_path)
    config = load_run_config(config_path) if config_path.exists() else build_bot_config()
    model = MaskablePPO.load(str(model_path))
    return model, config, config_path


def evaluate_on_slice(model, df_slice, feature_cols, config, stats_path, n_trials: int = 1,
                      n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Run one deterministic episode of ``model`` on ``df_slice`` and report.

    Returns a dict with the episode metrics, the statistical report, the final
    equity and the bar count. ``stats_path`` is the saved VecNormalize sidecar so
    observation scaling matches training; pass None to skip normalization.
    """
    if len(df_slice) <= config.env.window_size + 2:
        return {"skipped": True, "reason": "slice too short", "bars": int(len(df_slice))}

    vec_env = make_eval_env(df_slice, feature_cols, config, stats_path)
    model.set_env(vec_env)
    equity_curve, final_equity, closed_trades = run_one_episode(model, vec_env, deterministic=True)
    metrics = compute_metrics(closed_trades, equity_curve, bar_hours=config.env.bar_hours)
    stats = statistical_report(equity_curve, closed_trades, n_trials=n_trials,
                               n_resamples=n_resamples, seed=seed)
    return {
        "skipped": False,
        "bars": int(len(df_slice)),
        "final_equity": round(float(final_equity), 2),
        "n_trades": int(metrics["n_trades"]),
        "metrics": metrics,
        "statistics": stats,
    }


def assert_schema_or_raise(config_path, feature_cols):
    """Phase 2.4 guard reused by the holdout/regime evaluators."""
    assert_feature_schema(load_feature_columns(config_path), feature_cols)

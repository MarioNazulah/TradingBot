from __future__ import annotations

from pathlib import Path

# NOTE: stable_baselines3 is imported lazily inside make_train_env / make_eval_env
# (the only functions that wrap a VecEnv). make_env itself needs only the
# gym-based ForexTradingEnv, so importing it stays cheap and SB3-free — that is
# what lets baselines.py / regime_eval.py build a raw env without the heavy stack.
from trading_env import ForexTradingEnv


def make_env(df, feature_cols, config, random_start: bool = False, episode_max_steps: int | None = None):
    """Single source of truth for building a ForexTradingEnv from a BotConfig.

    Previously copy-pasted across train/walk_forward/eval scripts (review #4);
    parameter drift had already started. Keep all env wiring here.
    """
    env_cfg = config.env
    return ForexTradingEnv(
        df=df,
        window_size=env_cfg.window_size,
        sl_options=env_cfg.sl_options,
        tp_options=env_cfg.tp_options,
        spread_pips=env_cfg.spread_pips,
        commission_pips=env_cfg.commission_pips,
        max_slippage_pips=env_cfg.max_slippage_pips,
        random_start=random_start,
        min_episode_steps=env_cfg.min_episode_steps,
        episode_max_steps=episode_max_steps,
        feature_columns=feature_cols,
        hold_reward_weight=env_cfg.hold_reward_weight,
        open_penalty_pips=env_cfg.open_penalty_pips,
        time_penalty_pips=env_cfg.time_penalty_pips,
        unrealized_delta_weight=env_cfg.unrealized_delta_weight,
        allow_flip=env_cfg.allow_flip,
        action_space_mode=env_cfg.action_space_mode,
        reward_mode=env_cfg.reward_mode,
        commission_per_lot_usd=env_cfg.commission_per_lot_usd,
        swap_long_pips_per_day=env_cfg.swap_long_pips_per_day,
        swap_short_pips_per_day=env_cfg.swap_short_pips_per_day,
        variable_spread=env_cfg.variable_spread,
        news_spread_multiplier=env_cfg.news_spread_multiplier,
        swap_rollover_hour_utc=env_cfg.swap_rollover_hour_utc,
        bar_hours=env_cfg.bar_hours,
        fill_tiebreak_random_band=env_cfg.fill_tiebreak_random_band,
        max_lot_size_units=env_cfg.max_lot_size_units,
    )


def make_train_env(df, feature_cols, config, episode_max_steps: int | None = None):
    """Training vec env wrapped in VecNormalize (review #14).

    Features live on wildly different scales (RSI 0-100, ATR ~0.001, hour
    features in [-1, 1]); normalizing observations removes that mismatch.
    Rewards are already in pip units, so norm_reward stays off.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: make_env(df, feature_cols, config, True, episode_max_steps)])
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def make_eval_env(df, feature_cols, config, stats_path: str | Path | None = None):
    """Eval/test vec env. If a saved VecNormalize stat file exists, reuse the
    training statistics with updating disabled so eval matches training-time
    observation scaling exactly.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: make_env(df, feature_cols, config, False, None)])
    if stats_path is not None and Path(stats_path).exists():
        venv = VecNormalize.load(str(stats_path), venv)
        venv.training = False
        venv.norm_reward = False
    return venv


def make_callback_eval_env(df, feature_cols, config):
    """VecNormalize-wrapped eval env for SB3's (Maskable)EvalCallback.

    The callback calls ``sync_envs_normalization(train_env, eval_env)`` before
    every evaluation, so this env must itself be VecNormalize-wrapped for the
    training-time observation statistics to be copied in. ``training=False`` and
    ``norm_reward=False`` keep it from accumulating its own stats or distorting
    the (already pip-scaled) reward used for best-model selection.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: make_env(df, feature_cols, config, False, None)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.training = False
    return venv

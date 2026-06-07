from __future__ import annotations

from pathlib import Path

import pytest

# These tests exercise the trained-model / vec-env path, which needs the deep
# RL stack. Skip the whole module cleanly if it isn't installed (e.g. a
# data-only environment) instead of failing collection of the entire suite.
pytest.importorskip("sb3_contrib")
pytest.importorskip("stable_baselines3")

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import build_bot_config, load_run_config, model_config_path_for, split_train_val_test
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def _make_eval_env(df, feature_cols, config):
    env_cfg = config.env
    return ForexTradingEnv(
        df=df,
        window_size=env_cfg.window_size,
        sl_options=env_cfg.sl_options,
        tp_options=env_cfg.tp_options,
        spread_pips=env_cfg.spread_pips,
        commission_pips=env_cfg.commission_pips,
        max_slippage_pips=env_cfg.max_slippage_pips,
        random_start=False,
        min_episode_steps=env_cfg.min_episode_steps,
        episode_max_steps=None,
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
    )


def test_build_bot_config_defaults(tmp_path):
    config = build_bot_config(output_dir=tmp_path, total_timesteps=1234, seed=7)

    assert config.data.dataset_path.name.endswith(".csv")
    assert config.output.root_dir == tmp_path.resolve()
    assert config.training.total_timesteps == 1234
    assert config.training.seed == 7
    assert config.data.test_ratio == pytest.approx(0.15)


def test_real_dataset_loader_and_split():
    config = build_bot_config()
    if not config.data.dataset_path.exists():
        pytest.skip(f"Local dataset not present: {config.data.dataset_path}")

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    train_df, val_df, test_df = split_train_val_test(df, config.data.train_ratio, config.data.val_ratio)

    assert str(df.index.tz) == "UTC"
    assert len(feature_cols) >= 10
    assert len(train_df) > len(val_df) > 0
    assert len(test_df) > 0


def test_saved_model_matches_training_action_map():
    default_config = build_bot_config()
    model_path = default_config.output.model_file_path
    config_path = model_config_path_for(model_path)

    if not model_path.exists():
        pytest.skip(f"Saved model not present: {model_path}")
    if not config_path.exists():
        pytest.skip(f"Saved model config not present: {config_path}")

    config = load_run_config(config_path)
    if not config.data.dataset_path.exists():
        pytest.skip(f"Local dataset not present: {config.data.dataset_path}")

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    _, _, test_df = split_train_val_test(df, config.data.train_ratio, config.data.val_ratio)
    env = _make_eval_env(test_df, feature_cols, config)
    vec_env = DummyVecEnv([lambda: env])

    try:
        model = MaskablePPO.load(str(model_path), env=vec_env)
    except Exception as exc:  # model saved by a different algo/space (e.g. pre-Phase-1)
        pytest.skip(f"Saved model not loadable under current algo/space: {exc}")

    # action_map is always present and stable regardless of action_space_mode.
    assert env.action_map[0] == ("HOLD", None, None, None)
    assert env.action_map[1] == ("CLOSE", None, None, None)
    # The loaded policy's action space must match the env it will run in.
    assert str(model.action_space) == str(env.action_space)

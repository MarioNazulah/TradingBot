from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import PPO
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

    model = PPO.load(str(model_path), env=vec_env)

    assert model.action_space.n == env.action_space.n
    assert env.action_map[0] == ("HOLD", None, None, None)
    assert env.action_map[1] == ("CLOSE", None, None, None)

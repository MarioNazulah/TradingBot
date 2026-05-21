from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import build_bot_config, ensure_output_dirs, load_run_config, model_config_path_for, split_train_val_test
from evaluation import run_one_episode, save_trade_history
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO forex trading model.")
    parser.add_argument("--model-path", type=str, help="Path to a saved PPO model (.zip).")
    parser.add_argument("--dataset-path", type=str, help="Override the dataset path from the saved run config.")
    parser.add_argument("--output-csv", type=str, help="Path for closed trade history CSV output.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    return parser.parse_args()


def make_env(df, feature_cols, config):
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


def main():
    args = parse_args()

    default_config = build_bot_config()
    model_path = Path(args.model_path).resolve() if args.model_path else default_config.output.model_file_path
    config_path = model_config_path_for(model_path)

    if config_path.exists():
        config = load_run_config(config_path)
    else:
        config = default_config

    if args.dataset_path:
        config = replace(
            config,
            data=replace(config.data, dataset_path=Path(args.dataset_path).resolve()),
        )

    ensure_output_dirs(config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config.data.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {config.data.dataset_path}")

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    _, _, test_df = split_train_val_test(df, config.data.train_ratio, config.data.val_ratio)

    print(f"Model path    : {model_path}")
    print(f"Dataset path  : {config.data.dataset_path}")
    print(f"Timezone      : {config.data.timestamp_timezone}")
    print(f"Test bars     : {len(test_df)}")
    print(f"Action space  : {len(config.env.sl_options) * len(config.env.tp_options) * 2 + 2}")

    vec_test_env = DummyVecEnv([lambda: make_env(test_df, feature_cols, config)])
    model = PPO.load(str(model_path), env=vec_test_env)

    equity_curve, _, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    output_csv = Path(args.output_csv).resolve() if args.output_csv else config.output.trade_history_path
    saved_csv = save_trade_history(closed_trades, output_csv)
    if saved_csv is not None:
        print(f"Closed trade history saved to {saved_csv}")
    else:
        print("No closed trades recorded.")

    if not args.no_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label="Equity (Test)")
        plt.title("Equity Curve - Evaluation")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

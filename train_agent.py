from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from config import build_bot_config, ensure_output_dirs, save_json, save_run_config, split_train_val_test
from evaluation import compute_metrics, run_one_episode
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from utils.experiment_logger import log_experiment_to_obsidian


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PPO forex trading bot.")
    parser.add_argument("--dataset-path", type=str, help="Override the default training dataset path.")
    parser.add_argument("--output-dir", type=str, help="Directory for checkpoints, reports, and the saved model.")
    parser.add_argument("--timesteps", type=int, help="Override total PPO training timesteps.")
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    parser.add_argument("--enable-obsidian-log", action="store_true", help="Write experiment notes into the local Obsidian vault.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    return parser.parse_args()


def make_env(df, feature_cols, config, random_start: bool, episode_max_steps: int | None):
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
    )


def main():
    args = parse_args()
    config = build_bot_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        seed=args.seed,
        enable_obsidian_log=args.enable_obsidian_log,
    )

    ensure_output_dirs(config)
    set_random_seed(config.training.seed)

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    train_df, val_df, test_df = split_train_val_test(df, config.data.train_ratio, config.data.val_ratio)

    print(f"Dataset         : {config.data.dataset_path}")
    print(f"Timezone        : {config.data.timestamp_timezone}")
    print(f"Training bars   : {len(train_df)}")
    print(f"Validation bars : {len(val_df)}")
    print(f"Testing bars    : {len(test_df)}")
    print(f"Artifacts dir   : {config.output.root_dir}")
    print(f"Random seed     : {config.training.seed}")

    train_vec_env = DummyVecEnv([
        lambda: make_env(train_df, feature_cols, config, True, config.env.train_episode_max_steps)
    ])
    train_eval_env = DummyVecEnv([
        lambda: make_env(train_df, feature_cols, config, False, None)
    ])
    val_eval_env = DummyVecEnv([
        lambda: make_env(val_df, feature_cols, config, False, None)
    ])
    test_eval_env = DummyVecEnv([
        lambda: make_env(test_df, feature_cols, config, False, None)
    ])

    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=config.training.verbose,
        seed=config.training.seed,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        ent_coef=config.training.ent_coef,
        clip_range=config.training.clip_range,
        policy_kwargs={"net_arch": list(config.training.net_arch)},
        tensorboard_log=str(config.output.tensorboard_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.checkpoint_freq,
        save_path=str(config.output.checkpoints_dir),
        name_prefix=config.output.model_name,
    )

    model.learn(total_timesteps=config.training.total_timesteps, callback=checkpoint_callback)

    _, final_equity_val_last, _ = run_one_episode(model, val_eval_env)
    print(f"[VAL Eval] Last model final equity: {final_equity_val_last:.2f}")

    best_equity = -float("inf")
    best_path = None

    checkpoints = sorted(
        config.output.checkpoints_dir.glob(f"{config.output.model_name}*.zip"),
        key=lambda item: item.stat().st_mtime,
    )

    for checkpoint in checkpoints:
        try:
            candidate = PPO.load(str(checkpoint), env=val_eval_env)
            _, final_equity, _ = run_one_episode(candidate, val_eval_env)
            print(f"[VAL Eval] {checkpoint.name} -> final equity: {final_equity:.2f}")
            if final_equity > best_equity:
                best_equity = final_equity
                best_path = checkpoint
        except Exception as exc:
            print(f"[Skip] Could not evaluate checkpoint {checkpoint.name}: {exc}")

    if best_path is None or final_equity_val_last >= best_equity:
        print("Using last model as best (by validation final equity).")
        best_model = model
        best_equity = final_equity_val_last
    else:
        print(f"Using best checkpoint: {best_path} (validation final equity: {best_equity:.2f})")
        best_model = PPO.load(str(best_path), env=train_vec_env)

    best_model.save(str(config.output.model_base_path))
    save_run_config(config, config.output.model_config_path)
    save_run_config(config, config.output.run_config_path)
    print(f"Best model saved: {config.output.model_file_path}")
    print(f"Run config saved: {config.output.model_config_path}")

    equity_curve_train, final_equity_train, _ = run_one_episode(best_model, train_eval_env)
    equity_curve_val, final_equity_val, _ = run_one_episode(best_model, val_eval_env)
    equity_curve_test, final_equity_test, trades_test = run_one_episode(best_model, test_eval_env)

    print(f"[IS Eval]   Final equity (train): {final_equity_train:.2f}")
    print(f"[VAL Eval]  Final equity (val)  : {final_equity_val:.2f}")
    print(f"[OOS Eval]  Final equity (test) : {final_equity_test:.2f}")

    metrics = compute_metrics(trades_test, equity_curve_test)
    metrics_report = {
        **metrics,
        "final_equity_train": round(final_equity_train, 2),
        "final_equity_val": round(final_equity_val, 2),
        "final_equity_test": round(final_equity_test, 2),
        "best_validation_equity": round(best_equity, 2),
    }
    save_json(metrics_report, config.output.metrics_path)
    print(f"[OOS Metrics] Sharpe: {metrics['sharpe']} | Calmar: {metrics['calmar']} | "
          f"PF: {metrics['profit_factor']} | WR: {metrics['win_rate']} | "
          f"Trades: {metrics['n_trades']} | MaxDD: {metrics['max_dd']}")

    if config.output.enable_obsidian_log:
        log_experiment_to_obsidian(
            config={
                "n_actions": len(train_vec_env.get_attr("action_map")[0]),
                "sl_opts": list(config.env.sl_options),
                "tp_opts": list(config.env.tp_options),
                "window_size": config.env.window_size,
                "timesteps": config.training.total_timesteps,
                "dataset": str(config.data.dataset_path),
                "hold_reward_weight": config.env.hold_reward_weight,
                "open_penalty_pips": config.env.open_penalty_pips,
                "time_penalty_pips": config.env.time_penalty_pips,
                "seed": config.training.seed,
            },
            results=metrics_report,
            notes="",
            vault_path=config.output.obsidian_vault_dir,
        )

    if not args.no_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_train, label="Train (in-sample)")
        plt.plot(equity_curve_val, label="Validation")
        plt.plot(equity_curve_test, label="Test (out-of-sample)")
        plt.title("Equity Curves: Train vs Val vs Test (Best Model)")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

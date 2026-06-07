from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 30
    sl_options: tuple[int, ...] = (15, 30, 60)
    tp_options: tuple[int, ...] = (15, 30, 60)
    spread_pips: float = 1.0
    commission_pips: float = 0.0
    max_slippage_pips: float = 0.2
    hold_reward_weight: float = 0.01
    open_penalty_pips: float = 0.5
    time_penalty_pips: float = 0.005          # Phase 1.5: lowered from 0.02 so slow winners aren't net-penalized
    unrealized_delta_weight: float = 0.0
    min_episode_steps: int = 1000
    train_episode_max_steps: int = 2000
    allow_flip: bool = False
    # Phase 1.1: action space ("multidiscrete" default; "discrete" keeps the legacy Discrete(20) path)
    action_space_mode: str = "multidiscrete"
    # Phase 1.5: reward variant ("pnl" default; "r_multiple" = realized PnL in units of initial risk)
    reward_mode: str = "pnl"
    # Phase 1.3: realistic execution costs
    commission_per_lot_usd: float = 7.0       # round-turn ECN commission, charged at close
    swap_long_pips_per_day: float = -0.3      # EURUSD overnight financing (rate differential)
    swap_short_pips_per_day: float = 0.1
    variable_spread: tuple[float, float] | None = (0.8, 2.5)  # sampled per trade
    news_spread_multiplier: float = 3.0       # widen during overlap/news bars
    swap_rollover_hour_utc: int = 22
    bar_hours: float = 1.0
    # Phase 1.4: same-bar SL/TP fill ordering
    fill_tiebreak_random_band: float = 0.3
    # Hard cap on risk-based position size (review #3): a near-zero SL can
    # otherwise blow up lot size and nuke equity on a single trade.
    max_lot_size_units: float = 1_000_000.0


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path = PROJECT_ROOT / "data" / "EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    timestamp_timezone: str = "UTC"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # Phase 3.7: a strictly-newer second dataset, sliced from holdout_start_date,
    # is the never-touched final holdout. Used exactly once after all tuning.
    holdout_dataset_path: Path = (
        PROJECT_ROOT / "data" / "test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
    )
    holdout_start_date: str = "2024-01-01"

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path = PROJECT_ROOT / "artifacts"
    model_name: str = "model_eurusd_best"
    enable_obsidian_log: bool = True
    obsidian_vault_dir: Path = PROJECT_ROOT / "brain" / "tradingbot-brain" / "experiments"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root_dir / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.root_dir / "tensorboard"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    @property
    def model_base_path(self) -> Path:
        return self.root_dir / self.model_name

    @property
    def model_file_path(self) -> Path:
        return self.root_dir / f"{self.model_name}.zip"

    @property
    def model_config_path(self) -> Path:
        return self.root_dir / f"{self.model_name}_config.json"

    @property
    def vecnormalize_path(self) -> Path:
        return self.root_dir / f"{self.model_name}_vecnormalize.pkl"

    @property
    def run_config_path(self) -> Path:
        return self.root_dir / "latest_run_config.json"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "latest_metrics.json"

    @property
    def trade_history_path(self) -> Path:
        return self.reports_dir / "trade_history_output.csv"


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 60_000
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.01
    clip_range: float = 0.1
    net_arch: tuple[int, ...] = (256, 256)
    checkpoint_freq: int = 50_000
    seed: int = 42
    verbose: int = 1
    # Phase 3.1: EvalCallback evaluation cadence (env steps between val evals).
    eval_freq: int = 10_000
    # Phase 3.2: number of independent seeds per reported result (default 5).
    n_seeds: int = 5
    # PPO discount / GAE — surfaced so tune.py (3.4) can search them.
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass(frozen=True)
class BotConfig:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def build_bot_config(
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
    enable_obsidian_log: bool = True,
) -> BotConfig:
    base = BotConfig()

    data = DataConfig(
        dataset_path=Path(dataset_path).resolve() if dataset_path else base.data.dataset_path,
        timestamp_timezone=base.data.timestamp_timezone,
        train_ratio=base.data.train_ratio,
        val_ratio=base.data.val_ratio,
        holdout_dataset_path=base.data.holdout_dataset_path,
        holdout_start_date=base.data.holdout_start_date,
    )
    output = OutputConfig(
        root_dir=Path(output_dir).resolve() if output_dir else base.output.root_dir,
        model_name=base.output.model_name,
        enable_obsidian_log=enable_obsidian_log,
        obsidian_vault_dir=base.output.obsidian_vault_dir,
    )
    training = TrainingConfig(
        total_timesteps=total_timesteps if total_timesteps is not None else base.training.total_timesteps,
        n_steps=base.training.n_steps,
        batch_size=base.training.batch_size,
        ent_coef=base.training.ent_coef,
        clip_range=base.training.clip_range,
        net_arch=base.training.net_arch,
        checkpoint_freq=base.training.checkpoint_freq,
        seed=seed if seed is not None else base.training.seed,
        verbose=base.training.verbose,
        eval_freq=base.training.eval_freq,
        n_seeds=base.training.n_seeds,
        gamma=base.training.gamma,
        gae_lambda=base.training.gae_lambda,
    )
    return BotConfig(data=data, env=base.env, output=output, training=training)


def ensure_output_dirs(config: BotConfig) -> None:
    for directory in (
        config.output.root_dir,
        config.output.checkpoints_dir,
        config.output.tensorboard_dir,
        config.output.reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def split_train_val_test(df, train_ratio: float, val_ratio: float):
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must leave room for a test split.")

    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Split produced an empty train/validation/test segment.")

    return train_df, val_df, test_df


def model_config_path_for(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    stem = model_path.stem if model_path.suffix else model_path.name
    return model_path.with_name(f"{stem}_config.json")


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path


def save_run_config(
    config: BotConfig,
    path: str | Path,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Persist the run config sidecar.

    Phase 2.4: when ``feature_columns`` is provided, the ordered list is embedded
    under the top-level ``feature_columns`` key so a saved model can be checked
    against the live indicator output at eval time (catches silent indicator
    changes that would otherwise invalidate the model).
    """
    payload = _jsonify(asdict(config))
    if feature_columns is not None:
        payload["feature_columns"] = list(feature_columns)
    return save_json(payload, path)


def load_feature_columns(path: str | Path) -> list[str] | None:
    """Read the ordered feature-column schema from a saved sidecar, if present."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cols = raw.get("feature_columns")
    return list(cols) if cols is not None else None


def assert_feature_schema(saved: list[str] | None, live: list[str]) -> None:
    """Phase 2.4: fail-fast if the live feature schema drifted from the saved one.

    A no-op when no schema was saved (older models) so eval stays backward
    compatible, but raises on any ordering/content mismatch when one exists.
    """
    if saved is None:
        return
    if list(saved) != list(live):
        only_saved = [c for c in saved if c not in live]
        only_live = [c for c in live if c not in saved]
        raise ValueError(
            "Feature schema mismatch between saved model and live dataset.\n"
            f"  saved ({len(saved)}): {list(saved)}\n"
            f"  live  ({len(live)}): {list(live)}\n"
            f"  missing from live: {only_saved}\n"
            f"  unexpected in live: {only_live}\n"
            "The indicator pipeline changed since the model was trained; "
            "retrain or pin the indicators."
        )


def load_run_config(path: str | Path) -> BotConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    data = raw["data"]
    output = raw["output"]
    training = raw["training"]
    env = raw["env"]
    _D = EnvConfig()  # live defaults for any key missing from an old saved config

    return BotConfig(
        data=DataConfig(
            dataset_path=Path(data["dataset_path"]),
            timestamp_timezone=data["timestamp_timezone"],
            train_ratio=float(data["train_ratio"]),
            val_ratio=float(data["val_ratio"]),
            holdout_dataset_path=Path(data.get("holdout_dataset_path", DataConfig().holdout_dataset_path)),
            holdout_start_date=str(data.get("holdout_start_date", DataConfig().holdout_start_date)),
        ),
        # Every field reads through .get with the dataclass default so that
        # adding a new base EnvConfig field never breaks an already-saved
        # config (review #7). _D pulls the live default for each key.
        env=EnvConfig(
            window_size=int(env.get("window_size", _D.window_size)),
            sl_options=tuple(int(v) for v in env.get("sl_options", _D.sl_options)),
            tp_options=tuple(int(v) for v in env.get("tp_options", _D.tp_options)),
            spread_pips=float(env.get("spread_pips", _D.spread_pips)),
            commission_pips=float(env.get("commission_pips", _D.commission_pips)),
            max_slippage_pips=float(env.get("max_slippage_pips", _D.max_slippage_pips)),
            hold_reward_weight=float(env.get("hold_reward_weight", _D.hold_reward_weight)),
            open_penalty_pips=float(env.get("open_penalty_pips", _D.open_penalty_pips)),
            time_penalty_pips=float(env.get("time_penalty_pips", _D.time_penalty_pips)),
            unrealized_delta_weight=float(env.get("unrealized_delta_weight", _D.unrealized_delta_weight)),
            min_episode_steps=int(env.get("min_episode_steps", _D.min_episode_steps)),
            train_episode_max_steps=int(env.get("train_episode_max_steps", _D.train_episode_max_steps)),
            allow_flip=bool(env.get("allow_flip", _D.allow_flip)),
            action_space_mode=str(env.get("action_space_mode", "multidiscrete")),
            reward_mode=str(env.get("reward_mode", "pnl")),
            commission_per_lot_usd=float(env.get("commission_per_lot_usd", 7.0)),
            swap_long_pips_per_day=float(env.get("swap_long_pips_per_day", -0.3)),
            swap_short_pips_per_day=float(env.get("swap_short_pips_per_day", 0.1)),
            variable_spread=(
                tuple(float(v) for v in env["variable_spread"])
                if env.get("variable_spread") is not None else None
            ),
            news_spread_multiplier=float(env.get("news_spread_multiplier", 3.0)),
            swap_rollover_hour_utc=int(env.get("swap_rollover_hour_utc", 22)),
            bar_hours=float(env.get("bar_hours", 1.0)),
            fill_tiebreak_random_band=float(env.get("fill_tiebreak_random_band", 0.3)),
            max_lot_size_units=float(env.get("max_lot_size_units", _D.max_lot_size_units)),
        ),
        output=OutputConfig(
            root_dir=Path(output["root_dir"]),
            model_name=output["model_name"],
            enable_obsidian_log=bool(output["enable_obsidian_log"]),
            obsidian_vault_dir=Path(output["obsidian_vault_dir"]),
        ),
        training=TrainingConfig(
            total_timesteps=int(training["total_timesteps"]),
            n_steps=int(training["n_steps"]),
            batch_size=int(training["batch_size"]),
            ent_coef=float(training["ent_coef"]),
            clip_range=float(training["clip_range"]),
            net_arch=tuple(int(v) for v in training["net_arch"]),
            checkpoint_freq=int(training["checkpoint_freq"]),
            seed=int(training["seed"]),
            verbose=int(training["verbose"]),
            eval_freq=int(training.get("eval_freq", TrainingConfig().eval_freq)),
            n_seeds=int(training.get("n_seeds", TrainingConfig().n_seeds)),
            gamma=float(training.get("gamma", TrainingConfig().gamma)),
            gae_lambda=float(training.get("gae_lambda", TrainingConfig().gae_lambda)),
        ),
    )


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value

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
    time_penalty_pips: float = 0.02
    unrealized_delta_weight: float = 0.0
    min_episode_steps: int = 1000
    train_episode_max_steps: int = 2000
    allow_flip: bool = False


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path = PROJECT_ROOT / "data" / "EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    timestamp_timezone: str = "UTC"
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path = PROJECT_ROOT / "artifacts"
    model_name: str = "model_eurusd_best"
    enable_obsidian_log: bool = False
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
    enable_obsidian_log: bool = False,
) -> BotConfig:
    base = BotConfig()

    data = DataConfig(
        dataset_path=Path(dataset_path).resolve() if dataset_path else base.data.dataset_path,
        timestamp_timezone=base.data.timestamp_timezone,
        train_ratio=base.data.train_ratio,
        val_ratio=base.data.val_ratio,
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


def save_run_config(config: BotConfig, path: str | Path) -> Path:
    return save_json(_jsonify(asdict(config)), path)


def load_run_config(path: str | Path) -> BotConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    data = raw["data"]
    output = raw["output"]
    training = raw["training"]
    env = raw["env"]

    return BotConfig(
        data=DataConfig(
            dataset_path=Path(data["dataset_path"]),
            timestamp_timezone=data["timestamp_timezone"],
            train_ratio=float(data["train_ratio"]),
            val_ratio=float(data["val_ratio"]),
        ),
        env=EnvConfig(
            window_size=int(env["window_size"]),
            sl_options=tuple(int(v) for v in env["sl_options"]),
            tp_options=tuple(int(v) for v in env["tp_options"]),
            spread_pips=float(env["spread_pips"]),
            commission_pips=float(env["commission_pips"]),
            max_slippage_pips=float(env["max_slippage_pips"]),
            hold_reward_weight=float(env["hold_reward_weight"]),
            open_penalty_pips=float(env["open_penalty_pips"]),
            time_penalty_pips=float(env["time_penalty_pips"]),
            unrealized_delta_weight=float(env["unrealized_delta_weight"]),
            min_episode_steps=int(env["min_episode_steps"]),
            train_episode_max_steps=int(env["train_episode_max_steps"]),
            allow_flip=bool(env["allow_flip"]),
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

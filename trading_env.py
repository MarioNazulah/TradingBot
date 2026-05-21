# trading_env.py

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class ForexTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns=None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,
        commission_pips: float = 0.0,
        max_slippage_pips: float = 0.0,
        lot_size: float = 100000.0,
        reward_scale: float = 1.0,
        unrealized_delta_weight: float = 0.0,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        allow_flip: bool = False,
        hold_reward_weight: float = 0.0,
        open_penalty_pips: float = 0.5,
        time_penalty_pips: float = 0.02,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)
        else:
            self.feature_columns = list(feature_columns)

        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided.")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe is too short for the given window_size.")

        self.window_size = int(window_size)
        self.pip_value = float(pip_value)

        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.allow_flip = bool(allow_flip)

        # -----------------------------------------------
        # PRE-COMPUTE NUMPY ARRAYS (vectorized lookups)
        # All per-step data access goes through these —
        # 10-50x faster than df.loc inside the step loop
        # -----------------------------------------------
        self._close    = self.df["Close"].values.astype(np.float32)
        self._high     = self.df["High"].values.astype(np.float32)
        self._low      = self.df["Low"].values.astype(np.float32)
        self._features = self.df[self.feature_columns].values.astype(np.float32)

        # Action space
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation space — flat vector (Bug #3 fix)
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * self.base_num_features + self.state_num_features,),
            dtype=np.float32
        )

        self._reset_state()

    # ----------------------------
    # Core Helpers
    # ----------------------------

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.trade_high_water_mark = 0.0   # Bug #8 fix

        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd

        self.equity_curve = []
        self.last_trade_info = None

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 1000.0
        unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 100.0
        return np.array([pos, t_norm, unreal_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        # Vectorized: use numpy array instead of df.loc
        close_price = self._close[self.current_step]
        if self.position == 1:
            pnl_price = close_price - self.entry_price
        else:
            pnl_price = self.entry_price - close_price
        return pnl_price / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        # obs shape: (window_size * base_features + state_features,)
        # only normalize the window portion, not state features
        n_window = self.window_size * self.base_num_features
        obs[:n_window] = (obs[:n_window] - np.tile(self.feature_mean, self.window_size)) / np.tile(std, self.window_size)
        return obs

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)

        # Vectorized: slice numpy array directly (no DataFrame copy)
        base = self._features[start:self.current_step]

        if base.shape[0] == 0:
            base = np.tile(self._features[0], (self.window_size, 1))
        elif base.shape[0] < self.window_size:
            pad_rows = self.window_size - base.shape[0]
            pad = np.tile(base[0], (pad_rows, 1))
            base = np.vstack([pad, base])

        # Bug #3 fix: flatten window + append state as flat vector
        state_feat = self._get_state_features()
        obs = np.concatenate([base.flatten(), state_feat]).astype(np.float32)
        obs = self._apply_optional_normalization(obs)
        return obs

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        # Vectorized: use numpy array
        close_price = self._close[self.current_step]
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        # Bug #7 fix: risk-based position sizing (1% of equity per trade)
        risk_usd = self.equity_usd * 0.01
        self.lot_size = risk_usd / (sl_pips * self.pip_value * 100000.0)
        self.usd_per_pip = self.pip_value * self.lot_size * 100000.0

        if direction == 1:
            entry = close_price + slip_price
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:
            entry = close_price - slip_price
            sl_price = entry + sl_pips * self.pip_value
            tp_price = entry - tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.trade_high_water_mark = 0.0

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
        }

    def _close_position(self, reason: str, exit_price: float):
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips

        self.equity_usd += net_pips * self.usd_per_pip

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.trade_high_water_mark = 0.0   # Bug #8 fix

        self.last_trade_info = trade_info
        return net_pips

    def _check_sl_tp_intrabar_and_maybe_close(self):
        if self.position == 0:
            return None

        if self.current_step >= self.n_steps - 2:
            # Vectorized: use numpy array
            exit_price = self._close[self.current_step]
            return self._close_position("END_OF_DATA", exit_price)

        # Vectorized: use numpy arrays
        next_high = self._high[self.current_step + 1]
        next_low  = self._low[self.current_step + 1]

        if self.position == 1:
            sl_hit = next_low <= self.sl_price
            tp_hit = next_high >= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ----------------------------
    # Gym API
    # ----------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(np.random.randint(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()

        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1
        reward_pips = 0.0
        info = {}

        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]

        # 1) Action logic
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                # Vectorized
                close_price = self._close[self.current_step]
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)

        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                reward_pips -= self.open_penalty_pips
            else:
                if self.allow_flip:
                    # Vectorized
                    close_price = self._close[self.current_step]
                    reward_pips += self._close_position("FLIP_CLOSE", close_price)
                    self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                    reward_pips -= self.open_penalty_pips

        # 2) SL/TP check
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now

        # 3) Reward shaping while in position
        if self.position != 0:
            self.time_in_trade += 1
            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # Bug #8 fix: only reward NEW equity highs in the trade
            if unreal_now > self.trade_high_water_mark:
                reward_pips += self.hold_reward_weight * (unreal_now - self.trade_high_water_mark)
                self.trade_high_water_mark = unreal_now

            if self.unrealized_delta_weight != 0.0:
                reward_pips += self.unrealized_delta_weight * delta_unreal

            reward_pips -= self.time_penalty_pips
            self.prev_unrealized_pips = unreal_now

        # 4) Advance time
        self.current_step += 1

        # 5) Termination
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        # Drawdown-based early termination (70% of initial equity)
        if self.equity_usd < self.initial_equity_usd * 0.70:
            self.terminated = True

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # 6) Log equity
        self.equity_curve.append(float(self.equity_usd))

        # 7) Observation
        obs = self._get_observation()

        # 8) Scale reward
        reward = float(reward_pips) * self.reward_scale

        # 9) Info
        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "last_trade_info": self.last_trade_info,
        })

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, reward, done, info

    def render(self):
        print(
            f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )
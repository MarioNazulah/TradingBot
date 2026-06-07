# trading_env.py

from __future__ import annotations

import numpy as np
import pandas as pd

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
        lot_size_units: float = 100000.0,
        reward_scale: float = 1.0,
        unrealized_delta_weight: float = 0.0,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        allow_flip: bool = False,
        hold_reward_weight: float = 0.0,
        open_penalty_pips: float = 0.5,
        time_penalty_pips: float = 0.005,
        risk_per_trade: float = 0.01,
        # ---- Phase 1.1: action space ----
        action_space_mode: str = "multidiscrete",   # "multidiscrete" | "discrete"
        # ---- Phase 1.5: reward variant ----
        reward_mode: str = "pnl",                    # "pnl" | "r_multiple"
        # ---- Phase 1.3: realistic execution costs ----
        commission_per_lot_usd: float = 0.0,         # round-turn $/standard-lot, charged at close
        swap_long_pips_per_day: float = 0.0,
        swap_short_pips_per_day: float = 0.0,
        variable_spread: tuple[float, float] | None = None,  # (lo, hi) pips; None -> fixed spread_pips
        news_spread_multiplier: float = 1.0,
        swap_rollover_hour_utc: int = 22,
        bar_hours: float = 1.0,                       # used for swap accrual when no timestamps
        # ---- Phase 1.4: same-bar SL/TP fill order ----
        fill_tiebreak_random_band: float = 0.3,
        # Cap on risk-based lot size; guards against a near-zero SL producing
        # an enormous position that nukes equity on one trade.
        max_lot_size_units: float = 1_000_000.0,
    ):
        super().__init__()

        # Capture timestamps (if any) BEFORE dropping the index — needed for
        # swap accrual across the daily rollover (Phase 1.3).
        index = getattr(df, "index", None)
        if isinstance(index, pd.DatetimeIndex):
            # Normalise to naive UTC datetime64[ns] up front so the downstream
            # hour extraction for swap rollover (line ~295) is unambiguous and
            # never trips numpy's "no representation of timezones" warning,
            # regardless of the source index's tz.
            if index.tz is not None:
                index = index.tz_convert("UTC").tz_localize(None)
            self._timestamps = index.to_numpy()
        else:
            self._timestamps = None

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

        # Single source of truth: lot_size_units is the position size in base-currency units.
        # usd_per_pip is always derived from it. Both are recomputed on every OPEN.
        self.lot_size_units = float(lot_size_units)
        self.usd_per_pip = self.pip_value * self.lot_size_units

        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)
        self.risk_per_trade = float(risk_per_trade)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        self.allow_flip = bool(allow_flip)

        # ---- Phase 1 config ----
        self.action_space_mode = str(action_space_mode).lower()
        if self.action_space_mode not in ("multidiscrete", "discrete"):
            raise ValueError(f"Unknown action_space_mode: {action_space_mode!r}")
        self.reward_mode = str(reward_mode).lower()
        if self.reward_mode not in ("pnl", "r_multiple"):
            raise ValueError(f"Unknown reward_mode: {reward_mode!r}")

        self.commission_per_lot_usd = float(commission_per_lot_usd)
        self.swap_long_pips_per_day = float(swap_long_pips_per_day)
        self.swap_short_pips_per_day = float(swap_short_pips_per_day)
        self.variable_spread = tuple(variable_spread) if variable_spread is not None else None
        self.news_spread_multiplier = float(news_spread_multiplier)
        self.swap_rollover_hour_utc = int(swap_rollover_hour_utc)
        self.bar_hours = float(bar_hours)
        self.fill_tiebreak_random_band = float(fill_tiebreak_random_band)
        self.max_lot_size_units = float(max_lot_size_units)

        # -----------------------------------------------
        # PRE-COMPUTE NUMPY ARRAYS (vectorized lookups)
        # All per-step data access goes through these —
        # 10-50x faster than df.loc inside the step loop
        # -----------------------------------------------
        self._close    = self.df["Close"].values.astype(np.float32)
        self._high     = self.df["High"].values.astype(np.float32)
        self._low      = self.df["Low"].values.astype(np.float32)
        self._features = self.df[self.feature_columns].values.astype(np.float32)

        # Index of the London/NY overlap flag, used to widen variable spread
        # during the highest-volume window (a cheap proxy until a real news
        # calendar lands — see Phase 1.3).
        self._overlap_idx = (
            self.feature_columns.index("is_overlap")
            if "is_overlap" in self.feature_columns else None
        )

        # Action map (kind, direction, sl, tp). Always built: it backs the
        # legacy Discrete space AND decodes the sl/tp indices in MultiDiscrete.
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        if self.action_space_mode == "discrete":
            self.action_space = spaces.Discrete(len(self.action_map))
        else:
            # [structural(0=HOLD,1=OPEN,2=CLOSE), direction(0/1), sl_idx, tp_idx]
            self.action_space = spaces.MultiDiscrete(
                [3, 2, len(self.sl_options), len(self.tp_options)]
            )

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
        self.entry_step = None
        self.sl_price = None
        self.tp_price = None
        self.cur_sl_pips = None
        self.current_spread_pips = self.spread_pips
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

        # Flatten window + append state as flat vector. Normalization is the
        # caller's job (e.g. SB3 VecNormalize) — not done here. Wiring it in
        # the env would couple training-only stats into eval, so we keep this
        # raw and let the wrapper handle save/load.
        state_feat = self._get_state_features()
        obs = np.concatenate([base.flatten(), state_feat]).astype(np.float32)
        return obs

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        # Use the env's seeded RNG (set by gym.Env.reset(seed=...)). Going
        # through np.random.* leaks state across envs in the same process
        # and makes reset(seed=...) non-reproducible for slippage.
        return float(self.np_random.uniform(0.0, self.max_slippage_pips))

    def _is_news_bar(self) -> bool:
        # Proxy for elevated-spread conditions: the London/NY overlap window.
        # Replace with a real economic-calendar lookup later (Phase 1.3 note).
        if self._overlap_idx is None:
            return False
        return bool(self._features[self.current_step, self._overlap_idx] >= 0.5)

    def _sample_spread_pips(self) -> float:
        # Phase 1.3: per-trade variable spread, widened during news/overlap.
        if self.variable_spread is None:
            return self.spread_pips
        lo, hi = self.variable_spread
        spread = float(self.np_random.uniform(float(lo), float(hi)))
        if self._is_news_bar():
            spread *= self.news_spread_multiplier
        return spread

    def _cost_pips_round_trip(self) -> float:
        # Spread (variable, sampled at open) plus any fixed pip commission.
        return self.current_spread_pips + self.commission_pips

    def _compute_swap_pips(self) -> float:
        # Phase 1.3: swap/financing accrued while a position is held overnight.
        if self.position == 0 or self.entry_step is None:
            return 0.0
        per_day = self.swap_long_pips_per_day if self.position == 1 else self.swap_short_pips_per_day
        if per_day == 0.0:
            return 0.0
        if self._timestamps is not None:
            days = self._count_rollovers(self.entry_step, self.current_step)
        else:
            held_bars = max(0, self.current_step - self.entry_step)
            days = held_bars * self.bar_hours / 24.0
        return per_day * days

    def _count_rollovers(self, entry_step: int, exit_step: int) -> int:
        if self._timestamps is None or exit_step <= entry_step:
            return 0
        ts = self._timestamps[entry_step + 1: exit_step + 1]
        if len(ts) == 0:
            return 0
        # self._timestamps is already a numpy datetime64[ns] array (UTC).
        # Extract the UTC hour with pure numpy to avoid rebuilding a
        # pd.DatetimeIndex on every close (review #8).
        hours = (ts.astype("datetime64[h]").astype(np.int64)) % 24
        return int((hours == self.swap_rollover_hour_utc).sum())

    def _size_position(self, sl_pips: float) -> float:
        """Return position size in base-currency units so that a stop-out
        (including spread + worst-case slippage) loses at most
        risk_per_trade * equity. Single sizing path — no double scaling."""
        effective_sl = float(sl_pips) + self.current_spread_pips + self.max_slippage_pips
        if effective_sl <= 0:
            raise ValueError(f"Non-positive effective SL: {effective_sl}")
        risk_usd = self.equity_usd * self.risk_per_trade
        units = risk_usd / (effective_sl * self.pip_value)
        # Cap so a tiny SL can't produce a position large enough to wipe equity.
        return min(units, self.max_lot_size_units)

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        # Vectorized: use numpy array
        close_price = self._close[self.current_step]
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        # Sample the per-trade spread BEFORE sizing so risk uses real conditions.
        self.current_spread_pips = self._sample_spread_pips()

        # Risk-based sizing in base-currency units, costs included.
        self.lot_size_units = self._size_position(sl_pips)
        self.usd_per_pip = self.pip_value * self.lot_size_units

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
        self.entry_step = self.current_step
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.cur_sl_pips = float(sl_pips)
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
            "spread_pips": float(self.current_spread_pips),
        }

    def _close_position(self, reason: str, exit_price: float):
        # Invariant: usd_per_pip is always pip_value * lot_size_units.
        # Use an explicit raise (not assert) so the check survives `python -O`.
        if abs(self.usd_per_pip - self.pip_value * self.lot_size_units) >= 1e-9:
            raise RuntimeError(
                f"usd_per_pip ({self.usd_per_pip}) out of sync with "
                f"pip_value * lot_size_units ({self.pip_value * self.lot_size_units})"
            )

        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        cost_pips = self._cost_pips_round_trip()
        swap_pips = self._compute_swap_pips()
        net_pips = realized_pips - cost_pips + swap_pips

        # Commission is a USD amount per standard lot, charged at close.
        commission_usd = self.commission_per_lot_usd * (self.lot_size_units / 100000.0)

        self.equity_usd += net_pips * self.usd_per_pip - commission_usd

        # R-multiple of the closed trade (realized PnL in units of initial risk).
        sl_ref = self.cur_sl_pips
        realized_r = (net_pips / sl_ref) if sl_ref else 0.0
        self._last_realized_r = realized_r

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "swap_pips": float(swap_pips),
            "commission_usd": float(commission_usd),
            "net_pips": float(net_pips),
            "r_multiple": float(realized_r),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }

        self.position = 0
        self.entry_price = None
        self.entry_step = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.trade_high_water_mark = 0.0   # Bug #8 fix

        self.last_trade_info = trade_info
        return net_pips

    def _decide_sl_first(self, next_high: float, next_low: float) -> bool:
        # Phase 1.4: when SL and TP are both inside the next bar, decide which
        # fills first. Default to the level closer to entry; when the two are of
        # similar distance (within a band of the bar range), randomize weighted
        # by closeness so the policy cannot overfit a single deterministic rule.
        sl_dist = abs(self.entry_price - self.sl_price)
        tp_dist = abs(self.tp_price - self.entry_price)
        total = sl_dist + tp_dist
        if total <= 0:
            return True
        bar_range = float(next_high - next_low)
        if bar_range > 0 and abs(sl_dist - tp_dist) <= self.fill_tiebreak_random_band * bar_range:
            # Closer level (smaller distance) is more likely to fill first.
            prob_sl_first = tp_dist / total
            return float(self.np_random.uniform(0.0, 1.0)) < prob_sl_first
        return sl_dist <= tp_dist

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
        else:
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price

        if not sl_hit and not tp_hit:
            return None

        if sl_hit and tp_hit:
            sl_first = self._decide_sl_first(next_high, next_low)
        else:
            sl_first = bool(sl_hit)

        if sl_first:
            reason = "SL_AND_TP_SAME_BAR_SL_FIRST" if tp_hit else "SL_HIT"
            return self._close_position(reason, self.sl_price)
        else:
            reason = "SL_AND_TP_SAME_BAR_TP_FIRST" if sl_hit else "TP_HIT"
            return self._close_position(reason, self.tp_price)

    # ----------------------------
    # Action masking (Phase 1.2)
    # ----------------------------

    def action_masks(self) -> np.ndarray:
        """Boolean mask of currently-valid actions, consumed by MaskablePPO.

        - HOLD is always valid.
        - CLOSE is valid only while in a position.
        - OPEN is valid only while flat (or, with allow_flip, to reverse).

        Discrete mode: one bool per action_map entry.
        MultiDiscrete mode: flat concatenation of per-sub-space masks, length
        sum(nvec) = 3 + 2 + n_sl + n_tp.
        """
        flat = self.position == 0
        if self.action_space_mode == "discrete":
            mask = np.zeros(len(self.action_map), dtype=bool)
            for i, (kind, direction, _sl, _tp) in enumerate(self.action_map):
                if kind == "HOLD":
                    mask[i] = True
                elif kind == "CLOSE":
                    mask[i] = not flat
                elif kind == "OPEN":
                    if flat:
                        mask[i] = True
                    elif self.allow_flip:
                        cur_dir = 1 if self.position == 1 else 0
                        mask[i] = direction != cur_dir
            if not mask.any():
                mask[0] = True  # never leave the agent with no legal move
            return mask

        # MultiDiscrete
        structural = np.array([True, flat or self.allow_flip, not flat], dtype=bool)
        direction = np.array([True, True], dtype=bool)
        sl = np.ones(len(self.sl_options), dtype=bool)
        tp = np.ones(len(self.tp_options), dtype=bool)
        return np.concatenate([structural, direction, sl, tp])

    def _decode_action(self, action):
        """Map a raw action (int for Discrete, vector for MultiDiscrete) to
        (kind, direction, sl_pips, tp_pips)."""
        if self.action_space_mode == "discrete":
            return self.action_map[int(action)]

        a = np.asarray(action).reshape(-1)
        structural = int(a[0])
        if structural == 0:
            return ("HOLD", None, None, None)
        if structural == 2:
            return ("CLOSE", None, None, None)
        direction = int(a[1])  # 1 -> long, 0 -> short (matches action_map convention)
        sl_idx = int(np.clip(a[2], 0, len(self.sl_options) - 1))
        tp_idx = int(np.clip(a[3], 0, len(self.tp_options) - 1))
        return ("OPEN", direction, float(self.sl_options[sl_idx]), float(self.tp_options[tp_idx]))

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
                # Seeded RNG — see _sample_slippage_pips for rationale.
                self.current_step = int(self.np_random.integers(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()

        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1
        reward_pips = 0.0   # PnL-in-pips accounting (default reward)
        reward_r = 0.0      # R-multiple accounting (Phase 1.5, behind reward_mode)
        info = {}

        def _to_r(pips: float, sl_ref) -> float:
            return (pips / sl_ref) if sl_ref else 0.0

        act_type, direction, sl_pips, tp_pips = self._decode_action(action)

        # 1) Action logic
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                close_price = self._close[self.current_step]
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                sl_ref = self.cur_sl_pips
                realized = self._close_position("MANUAL_CLOSE", exit_price)
                reward_pips += realized
                reward_r += _to_r(realized, sl_ref)

        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                reward_pips -= self.open_penalty_pips
                reward_r -= _to_r(self.open_penalty_pips, self.cur_sl_pips)
            else:
                if self.allow_flip:
                    close_price = self._close[self.current_step]
                    sl_ref = self.cur_sl_pips
                    realized = self._close_position("FLIP_CLOSE", close_price)
                    reward_pips += realized
                    reward_r += _to_r(realized, sl_ref)
                    self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                    reward_pips -= self.open_penalty_pips
                    reward_r -= _to_r(self.open_penalty_pips, self.cur_sl_pips)

        # 2) SL/TP check
        sl_ref = self.cur_sl_pips
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now
            reward_r += _to_r(realized_now, sl_ref)

        # 3) Reward shaping while in position
        if self.position != 0:
            self.time_in_trade += 1
            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # Bug #8 fix: only reward NEW equity highs in the trade
            if unreal_now > self.trade_high_water_mark:
                bonus = self.hold_reward_weight * (unreal_now - self.trade_high_water_mark)
                reward_pips += bonus
                reward_r += _to_r(bonus, self.cur_sl_pips)
                self.trade_high_water_mark = unreal_now

            if self.unrealized_delta_weight != 0.0:
                shaped = self.unrealized_delta_weight * delta_unreal
                reward_pips += shaped
                reward_r += _to_r(shaped, self.cur_sl_pips)

            reward_pips -= self.time_penalty_pips
            reward_r -= _to_r(self.time_penalty_pips, self.cur_sl_pips)
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

        # 8) Select + scale reward
        base_reward = reward_r if self.reward_mode == "r_multiple" else reward_pips
        reward = float(base_reward) * self.reward_scale

        # 9) Info
        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "reward_r": float(reward_r),
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

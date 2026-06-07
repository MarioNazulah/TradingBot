# Code Review — ReinforcementTrading

**Verdict: Request Changes**

Good structure, solid test coverage, and clean env design. One critical runtime bug blocks evaluation entirely.

---

## Critical

1. **`_predict` undefined** (`evaluation.py:15`)
   `run_one_episode` calls `_predict(model, vec_env, obs, deterministic)` which doesn't exist — `NameError` on any evaluation call. Likely removed during a refactor. Replace with:
   ```python
   action, _ = model.predict(obs, deterministic=deterministic, action_masks=vec_env.env_method("action_masks"))
   ```

2. **`assert` as runtime invariant** (`trading_env.py:347`)
   `assert` is a no-op with `-O`. Swap for `if ... raise RuntimeError(...)`.

---

## High

3. **No lot size cap** (`trading_env.py:291–299`)
   `_size_position` can return enormous lot sizes when SL is near zero. One trade can nuke equity. Add a `max_lot_size` guard.

4. **`make_env` duplicated 3×** (`train_agent.py`, `walk_forward.py`, `test_agent.py`)
   Identical function copy-pasted. Move to a shared module — parameter drift is already visible in comments.

5. **Sharpe hardcoded to H1** (`evaluation.py:70`)
   `np.sqrt(252 * 24)` ignores `bar_hours`. Switch to `np.sqrt(252 * 24 / bar_hours)`.

6. **ATR division can produce `Inf`** (`indicators.py:39–48`)
   `dropna()` doesn't remove `Inf`. Add `df.replace([np.inf, -np.inf], np.nan, inplace=True)` before `dropna`.

7. **`load_run_config` breaks on new base fields** (`config.py:221–231`)
   Phase 1 fields use `.get()` with defaults; original fields don't. Adding any new base `EnvConfig` field breaks all saved configs.

---

## Medium

8. **`_count_rollovers` rebuilds `pd.DatetimeIndex` per close** (`trading_env.py:288`)
   `self._timestamps` is already numpy. Reconstruct from numpy ops to avoid repeated pandas overhead.

9. **Lambda closure over loop variables** (`walk_forward.py:136–144`)
   Works because `DummyVecEnv` calls lambdas immediately, but fragile. Use `lambda df=train_df: make_env(df, ...)`.

10. **Checkpoint selection by final equity alone** (`train_agent.py:136`)
    Final equity is path-dependent. Sharpe or multi-episode average would be a more stable selection signal.

11. **No OHLC consistency check** (`indicators.py`)
    Swapped columns or bad rows are silently corrupted. Validate `High >= Close >= Low` on load.

---

## Minor

12. `experiment_logger.py` — `datetime.now()` called twice; cache as `now = datetime.now()`.
13. `test_agent.py` — misleading name; pytest collects it as a test file. Rename to `eval_agent.py`.
14. No `VecNormalize` despite the env comment saying it's the caller's job. RSI (0–100) vs ATR (~0.001) vs hour features (-1, 1) — scale mismatch is likely hurting convergence at 60k steps.
15. `walk_forward.py` — val/test window sizes are relative to total dataset, not training segment. Early folds get val/test windows larger than their training set.

---

## What Looks Good

- Numpy pre-computation (`_close`, `_high`, `_low`, `_features`) — correct, avoids pandas in the step loop.
- Action masking correctly wired for MaskablePPO; MultiDiscrete structural sub-space is clean.
- `_decide_sl_first` probabilistic tiebreak prevents policy overfitting a deterministic fill rule.
- Risk-based sizing includes spread + slippage in effective SL — conservative and correct.
- Test coverage is solid: PnL, lookahead, SL/TP fill ordering, swap with timestamps, seeded determinism, risk cap.
- `frozen=True` dataclasses throughout `config.py` prevent accidental mutation.

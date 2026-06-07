"""Phase 3.7 — true holdout evaluation on the strictly-newer second dataset.

``data/test_EURUSD_..._20.02.2023-22.02.2025.csv`` overlaps the training file, so
only the part of it that is *newer than everything used during development* can
serve as an honest final test. We carve the slice from ``holdout_start_date``
(default 2024-01-01) onward and evaluate the chosen model on it exactly once.

"Once" is enforced softly: a marker file (``<artifacts>/reports/.holdout_used``)
records the model + dataset + timestamp of the first run. Re-running prints a loud
warning, because anything you look at more than once during tuning stops being a
holdout — re-tuning against this slice would re-introduce the selection bias the
holdout exists to measure.

    python holdout_eval.py --model-path artifacts/model_eurusd_best.zip
    python holdout_eval.py --holdout-start 2024-06-01 --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from config import build_bot_config, ensure_output_dirs, save_json
from indicators import load_and_preprocess_data
from model_eval import assert_schema_or_raise, evaluate_on_slice, load_model_and_config, slice_by_date


def parse_args():
    parser = argparse.ArgumentParser(description="Run the one-time true-holdout evaluation (Phase 3.7).")
    parser.add_argument("--model-path", type=str, help="Path to the saved model .zip (defaults to artifacts model).")
    parser.add_argument("--holdout-dataset", type=str, help="Override the holdout dataset path.")
    parser.add_argument("--holdout-start", type=str, help="Override holdout start date (e.g. 2024-01-01).")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of configurations evaluated before this model (feeds the Deflated Sharpe).")
    parser.add_argument("--force", action="store_true", help="Re-run even though the holdout was already used.")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path).resolve() if args.model_path else build_bot_config().output.model_file_path
    model, config, config_path = load_model_and_config(model_path)
    ensure_output_dirs(config)

    holdout_path = Path(args.holdout_dataset).resolve() if args.holdout_dataset else config.data.holdout_dataset_path
    holdout_start = args.holdout_start or config.data.holdout_start_date

    if not Path(holdout_path).exists():
        raise FileNotFoundError(f"Holdout dataset not found: {holdout_path}")

    marker = config.output.reports_dir / ".holdout_used"
    if marker.exists() and not args.force:
        prev = json.loads(marker.read_text(encoding="utf-8"))
        print("=" * 60)
        print("WARNING: this holdout has already been used.")
        print(f"  first used : {prev.get('used_at')}")
        print(f"  model      : {prev.get('model')}")
        print(f"  dataset    : {prev.get('dataset')} from {prev.get('holdout_start')}")
        print("Re-running re-introduces selection bias. Pass --force to proceed anyway.")
        print("=" * 60)
        return

    df, feature_cols = load_and_preprocess_data(holdout_path)
    # Fail-fast on indicator drift vs the trained model (Phase 2.4).
    assert_schema_or_raise(config_path, feature_cols)

    holdout_df = slice_by_date(df, start=holdout_start)
    print(f"Model           : {model_path}")
    print(f"Holdout dataset : {holdout_path}")
    print(f"Holdout window  : {holdout_start} onward  ({len(holdout_df)} bars)")

    result = evaluate_on_slice(
        model, holdout_df, feature_cols, config, config.output.vecnormalize_path,
        n_trials=args.n_trials,
    )

    if result.get("skipped"):
        print(f"Holdout slice too short to evaluate ({result.get('bars')} bars).")
        return

    m = result["metrics"]
    s = result["statistics"]
    print("\n" + "=" * 60)
    print("TRUE HOLDOUT RESULT (used once)")
    print("=" * 60)
    print(f"  Final equity : {result['final_equity']:.2f}")
    print(f"  Sharpe       : {m['sharpe']}   Calmar: {m['calmar']}   PF: {m['profit_factor']}")
    print(f"  Win rate     : {m['win_rate']}   Trades: {m['n_trades']}   MaxDD: {m['max_dd']}")
    print(f"  returns t-test (mean>0)   : t={s['returns_ttest']['t_stat']} p={s['returns_ttest']['p_value']}")
    print(f"  bootstrap Sharpe 95% CI   : {s['bootstrap_sharpe_ci']['sharpe']} "
          f"[{s['bootstrap_sharpe_ci']['ci_low']}, {s['bootstrap_sharpe_ci']['ci_high']}]")
    print(f"  deflated Sharpe (n={args.n_trials}) : DSR={s['deflated_sharpe']['dsr']} "
          f"SR0={s['deflated_sharpe']['sr0']}")

    report_path = config.output.reports_dir / "holdout_metrics.json"
    save_json({
        "model": str(model_path),
        "holdout_dataset": str(holdout_path),
        "holdout_start": holdout_start,
        "n_trials": args.n_trials,
        "result": result,
    }, report_path)
    print(f"\nReport saved: {report_path}")

    marker.write_text(json.dumps({
        "used_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "dataset": str(holdout_path),
        "holdout_start": holdout_start,
    }, indent=2), encoding="utf-8")
    print(f"Marked holdout as used: {marker}")


if __name__ == "__main__":
    main()

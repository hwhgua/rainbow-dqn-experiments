"""Validation and aggregation helpers for controlled DQN experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_EVALUATION_COLUMNS = {
    "run_id",
    "algorithm",
    "environment",
    "train_seed",
    "scheduled_env_step",
    "actual_env_step",
    "evaluation_round",
    "eval_episodes",
    "eval_seed_base",
    "eval_return_mean",
    "eval_return_std",
}


class ExperimentDataError(ValueError):
    """Raised when result files cannot support a fair comparison."""


def load_evaluation_summaries(experiment_dir):
    """Load every per-run evaluation summary under an experiment directory."""
    experiment_dir = Path(experiment_dir)
    paths = sorted(experiment_dir.rglob("evaluation_summary.csv"))
    if not paths:
        raise ExperimentDataError(f"找不到 evaluation_summary.csv：{experiment_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_EVALUATION_COLUMNS.difference(frame.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ExperimentDataError(f"{path} 缺少必要欄位：{missing_columns}")
        if frame.empty:
            raise ExperimentDataError(f"{path} 沒有 evaluation 記錄")
        frame = frame.copy()
        frame["source_path"] = str(path)
        frames.append(frame)

    summaries = pd.concat(frames, ignore_index=True)
    _validate_evaluation_coverage(summaries)
    return summaries


def _validate_evaluation_coverage(summaries):
    """Reject missing/duplicate evaluation points instead of silently filling."""
    key_columns = ["run_id", "algorithm", "train_seed", "scheduled_env_step"]
    duplicated = summaries.duplicated(key_columns, keep=False)
    if duplicated.any():
        rows = summaries.loc[duplicated, key_columns].to_dict("records")
        raise ExperimentDataError(f"evaluation summary 出現重複 step：{rows}")

    if summaries["eval_return_mean"].isna().any():
        raise ExperimentDataError("evaluation summary 含有空白的 eval_return_mean")

    step_sets = {}
    run_columns = ["run_id", "algorithm", "train_seed"]
    for run_key, frame in summaries.groupby(run_columns, sort=True):
        steps = tuple(sorted(frame["scheduled_env_step"].astype(int).tolist()))
        if not steps:
            raise ExperimentDataError(f"run 沒有 evaluation step：{run_key}")
        step_sets[run_key] = steps

    expected_steps = next(iter(step_sets.values()))
    mismatches = {
        str(run_key): steps
        for run_key, steps in step_sets.items()
        if steps != expected_steps
    }
    if mismatches:
        raise ExperimentDataError(
            "不同 run 的 evaluation step 不一致，拒絕插值或不公平彙整："
            f"expected={expected_steps}, mismatches={mismatches}"
        )


def aggregate_evaluations(summaries):
    """Aggregate per-seed evaluation means at each fixed training step."""
    _validate_evaluation_coverage(summaries)
    group_columns = ["algorithm", "environment", "scheduled_env_step"]
    rows = []

    for group_key, frame in summaries.groupby(group_columns, sort=True):
        values = frame["eval_return_mean"].astype(float).to_numpy()
        n_seeds = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n_seeds > 1 else 0.0
        sem = std / np.sqrt(n_seeds) if n_seeds > 0 else float("nan")
        rows.append(
            {
                "algorithm": group_key[0],
                "environment": group_key[1],
                "scheduled_env_step": int(group_key[2]),
                "n_seeds": int(n_seeds),
                "mean_eval_return": mean,
                "std_across_seed_means": std,
                "sem": float(sem),
                "ci95_low": float(mean - 1.96 * sem),
                "ci95_high": float(mean + 1.96 * sem),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["algorithm", "scheduled_env_step"], ignore_index=True
    )


def summarize_run_auc(summaries):
    """Return normalized evaluation AUC and final score for each training seed."""
    _validate_evaluation_coverage(summaries)
    rows = []
    run_columns = ["run_id", "algorithm", "environment", "train_seed"]

    for run_key, frame in summaries.groupby(run_columns, sort=True):
        frame = frame.sort_values("scheduled_env_step")
        steps = frame["scheduled_env_step"].astype(float).to_numpy()
        scores = frame["eval_return_mean"].astype(float).to_numpy()
        if len(steps) < 2 or steps[-1] <= steps[0]:
            normalized_auc = float("nan")
        else:
            normalized_auc = float(np.trapz(scores, steps) / (steps[-1] - steps[0]))

        rows.append(
            {
                "run_id": run_key[0],
                "algorithm": run_key[1],
                "environment": run_key[2],
                "train_seed": int(run_key[3]),
                "final_env_step": int(steps[-1]),
                "final_eval_return": float(scores[-1]),
                "normalized_eval_auc": normalized_auc,
            }
        )

    return pd.DataFrame(rows).sort_values(["algorithm", "train_seed"], ignore_index=True)


def write_aggregated_results(experiment_dir):
    """Persist the data tables consumed by the plotting CLI."""
    experiment_dir = Path(experiment_dir)
    summaries = load_evaluation_summaries(experiment_dir)
    aggregate = aggregate_evaluations(summaries)
    auc_summary = summarize_run_auc(summaries)

    aggregate_path = experiment_dir / "aggregate_evaluations.csv"
    auc_path = experiment_dir / "run_auc_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    auc_summary.to_csv(auc_path, index=False)
    return summaries, aggregate, auc_summary, aggregate_path, auc_path

"""Run a reproducible algorithm-by-seed DQN experiment matrix.

Jobs run sequentially on purpose: simultaneous PyTorch jobs compete for the
same GPU/CPU resources and make timing and memory failures harder to diagnose.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import SUPPORTED_ALGORITHMS
from experiment_results import write_aggregated_results
from train import train


JOB_COLUMNS = [
    "algorithm",
    "train_seed",
    "run_id",
    "status",
    "run_dir",
    "completed_env_steps",
    "gradient_updates",
    "final_eval_return",
    "error_type",
    "error_message",
]


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_experiment_spec(name, algorithms, seeds, total_env_steps, eval_every_steps, eval_episodes):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("--name 必須是單一、安全的資料夾名稱")
    if not algorithms:
        raise ValueError("至少需要一個演算法")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("--algorithms 不可重複")
    if not seeds:
        raise ValueError("至少需要一個 training seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds 不可重複")
    if total_env_steps <= 0:
        raise ValueError("--total-env-steps 必須大於 0")
    if eval_every_steps <= 0 or eval_every_steps >= total_env_steps:
        raise ValueError(
            "--eval-every-steps 必須大於 0 且小於 --total-env-steps，才能產生可比較曲線"
        )
    if eval_episodes <= 0:
        raise ValueError("--eval-episodes 必須大於 0")


def run_experiment(
    *,
    name,
    environment_name,
    algorithms,
    seeds,
    total_env_steps,
    eval_every_steps,
    eval_episodes,
    eval_seed_base,
    output_root="runs/experiments",
    fail_fast=False,
):
    """Run every requested ``(algorithm, training seed)`` pair sequentially."""
    algorithms = list(algorithms)
    seeds = [int(seed) for seed in seeds]
    _validate_experiment_spec(
        name, algorithms, seeds, total_env_steps, eval_every_steps, eval_episodes
    )

    experiment_dir = Path(output_root) / name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment_dir / "experiment_manifest.json"
    jobs_path = experiment_dir / "jobs_summary.csv"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_name,
        "algorithms": algorithms,
        "training_seeds": seeds,
        "total_env_steps": int(total_env_steps),
        "eval_every_steps": int(eval_every_steps),
        "eval_episodes": int(eval_episodes),
        "eval_seed_base": int(eval_seed_base),
        "evaluation_seed_policy": "eval_seed_base + evaluation_episode_index; reused for every round and job",
        "early_stopping": False,
        "execution_mode": "sequential",
        "versions": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "gymnasium": _package_version("gymnasium"),
            "highway_env": _package_version("highway-env"),
        },
        "git_commit": _git_commit(),
        "jobs": [],
    }
    _write_json(manifest_path, manifest)

    jobs = []
    for algorithm in algorithms:
        for seed in seeds:
            run_id = f"{algorithm}/seed_{seed}"
            job = {
                "algorithm": algorithm,
                "train_seed": seed,
                "run_id": run_id,
                "status": "running",
                "run_dir": None,
                "completed_env_steps": None,
                "gradient_updates": None,
                "final_eval_return": None,
                "error_type": None,
                "error_message": None,
            }
            jobs.append(job)
            print(f"\n{'#' * 72}\nExperiment job: {run_id}\n{'#' * 72}")

            try:
                run_paths = train(
                    environment_name,
                    algorithm,
                    seed=seed,
                    total_env_steps=total_env_steps,
                    eval_interval_steps=eval_every_steps,
                    eval_episodes=eval_episodes,
                    eval_seed_base=eval_seed_base,
                    output_root=experiment_dir,
                    run_id=run_id,
                    enable_early_stop=False,
                    experiment_name=name,
                )
                run_summary = json.loads(run_paths.summary_path.read_text(encoding="utf-8"))
                job.update(
                    {
                        "status": "completed",
                        "run_dir": str(run_paths.run_dir),
                        "completed_env_steps": run_summary["completed_env_steps"],
                        "gradient_updates": run_summary["gradient_updates"],
                        "final_eval_return": run_summary["final_eval_return"],
                    }
                )
            except Exception as error:  # Keep other algorithm/seed jobs auditable.
                job.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                print(f"\nJob failed: {run_id}: {type(error).__name__}: {error}")
                if fail_fast:
                    manifest["jobs"] = jobs
                    manifest["status"] = "failed"
                    _write_json(manifest_path, manifest)
                    pd.DataFrame(jobs, columns=JOB_COLUMNS).to_csv(jobs_path, index=False)
                    raise
            finally:
                manifest["jobs"] = jobs
                _write_json(manifest_path, manifest)
                pd.DataFrame(jobs, columns=JOB_COLUMNS).to_csv(jobs_path, index=False)

    failed_jobs = [job for job in jobs if job["status"] != "completed"]
    if not failed_jobs:
        _, _, _, aggregate_path, auc_path = write_aggregated_results(experiment_dir)
        manifest["aggregate_evaluations"] = str(aggregate_path)
        manifest["run_auc_summary"] = str(auc_path)
        manifest["status"] = "completed"
    else:
        manifest["status"] = "completed_with_failures"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["jobs"] = jobs
    _write_json(manifest_path, manifest)
    pd.DataFrame(jobs, columns=JOB_COLUMNS).to_csv(jobs_path, index=False)
    return experiment_dir, jobs


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="以固定 environment-step budget 執行 DQN variant × seed 實驗矩陣"
    )
    parser.add_argument("--name", required=True, help="本次實驗的唯一資料夾名稱")
    parser.add_argument("--env", choices=["Highway", "LunarLander"], default="Highway")
    parser.add_argument(
        "--algorithms", nargs="+", choices=SUPPORTED_ALGORITHMS, default=list(SUPPORTED_ALGORITHMS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--total-env-steps", required=True, type=int)
    parser.add_argument("--eval-every-steps", required=True, type=int)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-base", type=int, default=1_000_000)
    parser.add_argument("--output-root", default="runs/experiments")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    _, jobs = run_experiment(
        name=args.name,
        environment_name=args.env,
        algorithms=args.algorithms,
        seeds=args.seeds,
        total_env_steps=args.total_env_steps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_seed_base,
        output_root=args.output_root,
        fail_fast=args.fail_fast,
    )
    if any(job["status"] != "completed" for job in jobs):
        raise SystemExit(1)

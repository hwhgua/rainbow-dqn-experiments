"""Convenience entry point for one Highway training run."""

import argparse

from config import SUPPORTED_ALGORITHMS
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one DQN variant on Highway")
    parser.add_argument("--algo", choices=SUPPORTED_ALGORITHMS, default="Rainbow")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed-base", type=int, default=1_000_000)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    early_stop_group = parser.add_mutually_exclusive_group()
    early_stop_group.add_argument(
        "--enable-early-stop", dest="enable_early_stop", action="store_true"
    )
    early_stop_group.add_argument(
        "--disable-early-stop", dest="enable_early_stop", action="store_false"
    )
    parser.set_defaults(enable_early_stop=None)
    args = parser.parse_args()
    if args.episodes is not None and args.total_env_steps is not None:
        parser.error("--episodes 與 --total-env-steps 不能同時使用")
    train(
        "Highway",
        args.algo,
        args.episodes,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        eval_interval_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_seed_base,
        output_root=args.output_root,
        run_id=args.run_id,
        enable_early_stop=args.enable_early_stop,
        experiment_name=args.experiment_name,
    )

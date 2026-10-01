"""Shared training loop for the implemented DQN variants.

The module supports both interactive episode-based training and controlled
experiments with a fixed number of training environment interactions.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import numpy as np
import pandas as pd

from agent_factory import build_agent
from checkpointing import (
    load_checkpoint,
    restore_model_weights,
    save_model_checkpoint,
    validate_checkpoint_metadata,
)
from config import Config, SUPPORTED_ALGORITHMS, create_run_paths, write_run_config
from environment import get_environment_snapshot, make_environment
from evaluate import (
    EVALUATION_EPISODE_COLUMNS,
    EVALUATION_SUMMARY_COLUMNS,
    evaluate_policy,
)
from utils import set_global_seed


TRAINING_LOG_COLUMNS = [
    "run_id",
    "algorithm",
    "environment",
    "train_seed",
    "episode",
    "global_step_start",
    "global_step_end",
    "episode_steps",
    "score",
    "avg_score",
    "exploration_strategy",
    "epsilon_greedy",
    "epsilon_start",
    "epsilon_end",
    "terminated",
    "truncated",
    "budget_cutoff",
    "max_t_cutoff",
    "replay_size",
    "gradient_updates",
]


def _write_csv(records, path, columns):
    """Write a schema-stable CSV, including headers when no records exist."""
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


def _write_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _epsilon_for_step(config, global_step, legacy_epsilon):
    """Return an exploration rate for the current training interaction."""
    anneal_steps = config.EPS_ANNEAL_STEPS
    if anneal_steps is None:
        return legacy_epsilon

    progress = min(1.0, global_step / anneal_steps)
    return config.EPS_START + progress * (config.EPS_END - config.EPS_START)


def train(
    environment_name="Highway",
    algorithm_name="Rainbow",
    n_episodes=None,
    *,
    seed=None,
    total_env_steps=None,
    eval_interval_steps=None,
    eval_episodes=5,
    eval_seed_base=1_000_000,
    output_root="runs",
    run_id=None,
    enable_early_stop=None,
    experiment_name=None,
):
    """Train one agent and return the paths created for this run.

    ``total_env_steps`` counts interactions with the *training* environment
    only. When it is supplied, epsilon and PER beta anneal by global step and
    early stopping defaults to off, so every experiment job receives the same
    training budget.
    """
    if n_episodes is not None and n_episodes <= 0:
        raise ValueError("n_episodes 必須大於 0")
    if total_env_steps is not None and total_env_steps <= 0:
        raise ValueError("total_env_steps 必須大於 0")
    if eval_interval_steps is not None and eval_interval_steps <= 0:
        raise ValueError("eval_interval_steps 必須大於 0")
    if eval_interval_steps is not None and eval_episodes <= 0:
        raise ValueError("eval_episodes 必須大於 0")

    cfg = Config(environment_name, algorithm_name)
    if seed is not None:
        cfg.SEED = int(seed)
    if n_episodes is not None:
        cfg.N_EPISODES = int(n_episodes)

    cfg.TOTAL_ENV_STEPS = int(total_env_steps) if total_env_steps is not None else None
    cfg.EVAL_INTERVAL_STEPS = (
        int(eval_interval_steps) if eval_interval_steps is not None else None
    )
    cfg.EVAL_EPISODES = int(eval_episodes) if eval_interval_steps is not None else None
    cfg.EVAL_SEED_BASE = int(eval_seed_base) if eval_interval_steps is not None else None
    cfg.EXPERIMENT_NAME = experiment_name
    cfg.ENABLE_EARLY_STOP = (
        total_env_steps is None if enable_early_stop is None else bool(enable_early_stop)
    )

    # A fixed step budget needs schedules that do not depend on each model's
    # episode lengths. The default interactive path retains its old behavior.
    if total_env_steps is not None:
        cfg.EPS_ANNEAL_STEPS = int(total_env_steps)
        cfg.PER_BETA_ANNEAL_STEPS = int(total_env_steps)

    run_paths = create_run_paths(cfg, output_root=output_root, run_id=run_id)
    run_identifier = (
        str(run_id).replace("\\", "/") if run_id is not None else run_paths.run_dir.name
    )
    write_run_config(run_paths, cfg, get_environment_snapshot(cfg.ENV_NAME))
    set_global_seed(cfg.SEED)

    render_mode = "human" if cfg.RENDER else None
    env = make_environment(cfg.ENV_NAME, render_mode=render_mode)
    env.action_space.seed(cfg.SEED)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = build_agent(cfg.ALGO_NAME, state_size, action_size, cfg)
    uses_epsilon = agent.uses_epsilon_greedy

    print(f"\n{'=' * 40}")
    print(f"Training Model: {cfg.ALGO_NAME}")
    print(f"Environment: {cfg.ENV_NAME}")
    print(f"Training seed: {cfg.SEED}")
    print(f"State Size: {state_size}, Action Size: {action_size}")
    if total_env_steps is not None:
        print(f"Training budget: {total_env_steps} environment steps")
    print(f"Run directory: {run_paths.run_dir}")
    print(f"{'=' * 40}\n")

    if cfg.RESUME_TRAIN:
        if not cfg.RESUME_PATH:
            raise ValueError("RESUME_TRAIN=True 時必須在 config.py 提供 RESUME_PATH")
        payload = load_checkpoint(cfg.RESUME_PATH, map_location="cpu")
        if payload["metadata"] is None:
            print("警告：載入的是舊式純權重 checkpoint；只能 warm-start，無法驗證原始設定。")
        else:
            validate_checkpoint_metadata(
                payload,
                algorithm=cfg.ALGO_NAME,
                environment=cfg.ENV_NAME,
                state_size=state_size,
                action_size=action_size,
            )
        restore_model_weights(agent, payload)
        print(f"已載入模型權重作為 warm start：{cfg.RESUME_PATH}")

    legacy_epsilon = cfg.EPS_END if cfg.RESUME_TRAIN and uses_epsilon else (
        cfg.EPS_START if uses_epsilon else None
    )
    scores_window = deque(maxlen=100)
    training_history = []
    evaluation_episode_history = []
    evaluation_summary_history = []
    best_training_score = -np.inf
    best_evaluation_score = None
    patience_counter = 0
    patience_limit = 100
    global_step = 0
    last_episode = cfg.START_EPISODE - 1
    last_average_score = float("nan")
    evaluation_round = 0
    started_at = time.perf_counter()

    def checkpoint_extra():
        return {
            "requested_total_env_steps": cfg.TOTAL_ENV_STEPS,
            "gradient_updates": int(agent.gradient_updates),
            "evaluation_round": int(evaluation_round),
            "best_eval_return": best_evaluation_score,
        }

    def save_checkpoint(path, episode, average_score):
        save_model_checkpoint(
            path,
            agent,
            cfg,
            state_size,
            action_size,
            episode=episode,
            average_score=average_score,
            global_step=global_step,
            training_state_extra=checkpoint_extra(),
        )

    def record_evaluation(current_episode, current_average_score):
        nonlocal best_evaluation_score, evaluation_round
        evaluation_round += 1
        evaluation_started_at = time.perf_counter()
        records, summary = evaluate_policy(
            agent,
            cfg,
            episodes=cfg.EVAL_EPISODES,
            seed_base=cfg.EVAL_SEED_BASE,
            scheduled_env_step=global_step,
            actual_env_step=global_step,
            evaluation_round=evaluation_round,
            run_id=run_identifier,
            render=False,
        )
        summary["elapsed_seconds"] = round(time.perf_counter() - evaluation_started_at, 6)
        evaluation_episode_history.extend(records)
        evaluation_summary_history.append(summary)
        _write_csv(
            evaluation_episode_history,
            run_paths.evaluation_episode_log_path,
            EVALUATION_EPISODE_COLUMNS,
        )
        _write_csv(
            evaluation_summary_history,
            run_paths.evaluation_summary_log_path,
            EVALUATION_SUMMARY_COLUMNS,
        )

        if best_evaluation_score is None or summary["eval_return_mean"] > best_evaluation_score:
            best_evaluation_score = summary["eval_return_mean"]
            save_checkpoint(run_paths.best_model_path, current_episode, current_average_score)
            print(
                f"\nEvaluation at step {global_step}: "
                f"{best_evaluation_score:.2f} (best checkpoint updated)"
            )
        else:
            print(
                f"\nEvaluation at step {global_step}: "
                f"{summary['eval_return_mean']:.2f}"
            )

    # When a total step budget is provided without an explicit episode limit,
    # do not silently stop at Config.N_EPISODES before reaching the budget.
    episode_limit = cfg.N_EPISODES if total_env_steps is None or n_episodes is not None else None
    i_episode = cfg.START_EPISODE

    try:
        while episode_limit is None or i_episode <= episode_limit:
            if total_env_steps is not None and global_step >= total_env_steps:
                break

            state, _ = env.reset(seed=cfg.SEED + i_episode - cfg.START_EPISODE)
            episode_start_step = global_step
            score = 0.0
            episode_steps = 0
            episode_terminated = False
            episode_truncated = False
            budget_cutoff = False
            max_t_cutoff = False
            epsilon_start = None
            epsilon_end = None

            for local_step in range(1, cfg.MAX_T + 1):
                action_epsilon = (
                    _epsilon_for_step(cfg, global_step, legacy_epsilon)
                    if uses_epsilon
                    else 0.0
                )
                if uses_epsilon and epsilon_start is None:
                    epsilon_start = action_epsilon
                if uses_epsilon:
                    epsilon_end = action_epsilon

                action = agent.act(state, eps=action_epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                will_exhaust_budget = (
                    total_env_steps is not None and global_step + 1 >= total_env_steps
                )
                reaches_local_horizon = local_step == cfg.MAX_T
                episode_end = terminated or truncated or will_exhaust_budget or reaches_local_horizon

                # A budget boundary is an episode boundary for N-step flushing,
                # but it is not a Bellman terminal state: bootstrap remains valid.
                agent.step(
                    state,
                    action,
                    reward,
                    next_state,
                    done=terminated,
                    episode_end=episode_end,
                )
                state = next_state
                score += reward
                global_step += 1
                episode_steps += 1
                episode_terminated = terminated
                episode_truncated = truncated
                budget_cutoff = will_exhaust_budget and not (terminated or truncated)
                max_t_cutoff = reaches_local_horizon and not (terminated or truncated)

                if eval_interval_steps is not None and global_step % eval_interval_steps == 0:
                    record_evaluation(
                        i_episode,
                        float(np.mean(scores_window)) if scores_window else float("nan"),
                    )

                if episode_end:
                    break

            scores_window.append(score)
            current_avg_score = float(np.mean(scores_window))
            last_episode = i_episode
            last_average_score = current_avg_score
            training_history.append(
                {
                    "run_id": run_identifier,
                    "algorithm": cfg.ALGO_NAME,
                    "environment": cfg.ENV_NAME,
                    "train_seed": int(cfg.SEED),
                    "episode": i_episode,
                    "global_step_start": episode_start_step,
                    "global_step_end": global_step,
                    "episode_steps": episode_steps,
                    "score": score,
                    "avg_score": current_avg_score,
                    "exploration_strategy": agent.exploration_strategy,
                    "epsilon_greedy": epsilon_start if uses_epsilon else np.nan,
                    "epsilon_start": epsilon_start if uses_epsilon else np.nan,
                    "epsilon_end": epsilon_end if uses_epsilon else np.nan,
                    "terminated": episode_terminated,
                    "truncated": episode_truncated,
                    "budget_cutoff": budget_cutoff,
                    "max_t_cutoff": max_t_cutoff,
                    "replay_size": len(agent.memory),
                    "gradient_updates": int(agent.gradient_updates),
                }
            )

            print(f"\rEpisode {i_episode}\tAverage Score: {current_avg_score:.2f}", end="")

            if i_episode % 10 == 0:
                _write_csv(training_history, run_paths.log_path, TRAINING_LOG_COLUMNS)

            # Retain the historical interactive behavior when no scheduled
            # deterministic evaluation is requested. Experiment jobs choose
            # the best model exclusively by evaluation mean instead.
            if eval_interval_steps is None and i_episode > 100:
                if current_avg_score > 0 and current_avg_score > best_training_score:
                    best_training_score = current_avg_score
                    save_checkpoint(run_paths.best_model_path, i_episode, current_avg_score)
                    print(f"\nbest training average: ({best_training_score:.2f}), checkpoint updated.")
                    patience_counter = 0
                else:
                    patience_counter += 1

            if uses_epsilon and cfg.EPS_ANNEAL_STEPS is None:
                legacy_epsilon = max(cfg.EPS_END, cfg.EPS_DECAY * legacy_epsilon)

            if (
                cfg.ENABLE_EARLY_STOP
                and eval_interval_steps is None
                and i_episode > 100
                and current_avg_score > cfg.EARLY_STOP_SCORE
                and patience_counter >= patience_limit
            ):
                print(f"\n\n模型已連續 {patience_limit} 回合未突破最佳分數 ({best_training_score:.2f})。")
                print("依目前設定提前結束訓練。")
                break

            if total_env_steps is not None and global_step >= total_env_steps:
                break
            i_episode += 1

        # Always record the final point, including a non-divisible step budget.
        if (
            eval_interval_steps is not None
            and global_step > 0
            and (
                not evaluation_summary_history
                or evaluation_summary_history[-1]["actual_env_step"] != global_step
            )
        ):
            record_evaluation(last_episode, last_average_score)

        _write_csv(training_history, run_paths.log_path, TRAINING_LOG_COLUMNS)
        _write_csv(
            evaluation_episode_history,
            run_paths.evaluation_episode_log_path,
            EVALUATION_EPISODE_COLUMNS,
        )
        _write_csv(
            evaluation_summary_history,
            run_paths.evaluation_summary_log_path,
            EVALUATION_SUMMARY_COLUMNS,
        )
        save_checkpoint(run_paths.final_model_path, last_episode, last_average_score)

        final_evaluation = evaluation_summary_history[-1] if evaluation_summary_history else None
        run_summary = {
            "status": "completed",
            "run_id": run_identifier,
            "algorithm": cfg.ALGO_NAME,
            "environment": cfg.ENV_NAME,
            "train_seed": int(cfg.SEED),
            "requested_total_env_steps": cfg.TOTAL_ENV_STEPS,
            "completed_env_steps": int(global_step),
            "episodes_completed": int(len(training_history)),
            "gradient_updates": int(agent.gradient_updates),
            "evaluation_rounds": int(evaluation_round),
            "best_eval_return": best_evaluation_score,
            "final_eval_return": (
                final_evaluation["eval_return_mean"] if final_evaluation is not None else None
            ),
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
            "paths": {
                "training_log": str(run_paths.log_path),
                "evaluation_summary": str(run_paths.evaluation_summary_log_path),
                "final_model": str(run_paths.final_model_path),
            },
        }
        _write_json(run_summary, run_paths.summary_path)
    finally:
        env.close()

    print(f"\nFinal model saved to: {run_paths.final_model_path}")
    return run_paths


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="訓練 DQN variants 或 Rainbow DQN")
    parser.add_argument("--env", choices=["Highway", "LunarLander"], default="Highway")
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
    return parser


if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()
    if args.episodes is not None and args.total_env_steps is not None:
        parser.error("--episodes 與 --total-env-steps 不能同時使用")
    train(
        args.env,
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

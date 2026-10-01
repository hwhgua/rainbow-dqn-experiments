"""Evaluate a metadata checkpoint or a legacy model state_dict."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agent_factory import AGENT_MAPPING, build_agent
from checkpointing import load_checkpoint, restore_model_weights, validate_checkpoint_metadata
from config import Config
from environment import make_environment
from utils import set_global_seed


EVALUATION_EPISODE_COLUMNS = [
    "run_id",
    "algorithm",
    "environment",
    "train_seed",
    "scheduled_env_step",
    "actual_env_step",
    "evaluation_round",
    "evaluation_episode",
    "evaluation_seed",
    "return",
    "episode_steps",
    "terminated",
    "truncated",
    "deterministic",
]

EVALUATION_SUMMARY_COLUMNS = [
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
    "eval_return_min",
    "eval_return_max",
    "eval_episode_length_mean",
    "elapsed_seconds",
]


def _resolve_runtime_config(payload, algorithm_name=None, environment_name=None):
    metadata = payload.get("metadata")
    if metadata is None:
        if algorithm_name is None or environment_name is None:
            raise ValueError(
                "舊式純權重 checkpoint 缺少 metadata；請同時提供 --algo 與 --env。"
            )
        print("警告：舊式 checkpoint 無法驗證原始演算法、環境、seed 與超參數。")
        return Config(environment_name, algorithm_name), algorithm_name, environment_name

    checkpoint_algorithm = metadata.get("algorithm")
    checkpoint_environment = metadata.get("environment")
    if not checkpoint_algorithm or not checkpoint_environment:
        raise ValueError("checkpoint metadata 缺少 algorithm 或 environment")
    if algorithm_name is not None and algorithm_name != checkpoint_algorithm:
        raise ValueError(
            f"--algo={algorithm_name!r} 與 checkpoint 的 {checkpoint_algorithm!r} 不一致。"
        )
    if environment_name is not None and environment_name != checkpoint_environment:
        raise ValueError(
            f"--env={environment_name!r} 與 checkpoint 的 {checkpoint_environment!r} 不一致。"
        )

    config_snapshot = metadata.get("config")
    if config_snapshot is None:
        raise ValueError("checkpoint metadata 缺少 config snapshot")
    config = Config.from_snapshot(config_snapshot)
    if config.ALGO_NAME != checkpoint_algorithm or config.ENV_NAME != checkpoint_environment:
        raise ValueError("checkpoint metadata 的 config snapshot 與頂層 algorithm/environment 不一致")
    return config, checkpoint_algorithm, checkpoint_environment


def evaluate_policy(
    agent,
    config,
    *,
    episodes,
    seed_base,
    scheduled_env_step=0,
    actual_env_step=None,
    evaluation_round=0,
    run_id="standalone",
    render=False,
):
    """Evaluate an in-memory agent with fixed, deterministic test episodes.

    This helper deliberately does not call ``set_global_seed``: doing so in
    the middle of training would reset the training RNG stream.  Instead it
    seeds only an independent evaluation environment with the same seed list
    on every evaluation round and every algorithm run.
    """
    if episodes <= 0:
        raise ValueError("episodes 必須大於 0")
    if actual_env_step is None:
        actual_env_step = scheduled_env_step

    env = make_environment(config.ENV_NAME, render_mode="human" if render else None)
    env.action_space.seed(seed_base)
    was_training = agent.qnetwork_local.training
    records = []

    try:
        for evaluation_episode in range(1, episodes + 1):
            evaluation_seed = seed_base + evaluation_episode - 1
            state, _ = env.reset(seed=evaluation_seed)
            score = 0.0
            episode_steps = 0
            terminated = False
            truncated = False

            for _ in range(config.MAX_T):
                action = agent.act(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                score += reward
                episode_steps += 1
                if terminated or truncated:
                    break

            records.append(
                {
                    "run_id": run_id,
                    "algorithm": config.ALGO_NAME,
                    "environment": config.ENV_NAME,
                    "train_seed": int(config.SEED),
                    "scheduled_env_step": int(scheduled_env_step),
                    "actual_env_step": int(actual_env_step),
                    "evaluation_round": int(evaluation_round),
                    "evaluation_episode": evaluation_episode,
                    "evaluation_seed": int(evaluation_seed),
                    "return": float(score),
                    "episode_steps": episode_steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "deterministic": True,
                }
            )
    finally:
        env.close()
        agent.qnetwork_local.train(was_training)

    scores = np.asarray([record["return"] for record in records], dtype=float)
    lengths = np.asarray([record["episode_steps"] for record in records], dtype=float)
    summary = {
        "run_id": run_id,
        "algorithm": config.ALGO_NAME,
        "environment": config.ENV_NAME,
        "train_seed": int(config.SEED),
        "scheduled_env_step": int(scheduled_env_step),
        "actual_env_step": int(actual_env_step),
        "evaluation_round": int(evaluation_round),
        "eval_episodes": int(episodes),
        "eval_seed_base": int(seed_base),
        "eval_return_mean": float(scores.mean()),
        "eval_return_std": float(scores.std(ddof=0)),
        "eval_return_min": float(scores.min()),
        "eval_return_max": float(scores.max()),
        "eval_episode_length_mean": float(lengths.mean()),
    }
    return records, summary


def evaluate(model_path, algorithm_name=None, environment_name=None, episodes=5, render=True):
    """Evaluate a checkpoint with a deterministic greedy policy."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重：{model_path}")
    if episodes <= 0:
        raise ValueError("episodes 必須大於 0")

    payload = load_checkpoint(model_path, map_location="cpu")
    cfg, algorithm_name, environment_name = _resolve_runtime_config(
        payload, algorithm_name, environment_name
    )
    set_global_seed(cfg.SEED)

    env = make_environment(cfg.ENV_NAME, render_mode="human" if render else None)
    env.action_space.seed(cfg.SEED)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    validate_checkpoint_metadata(
        payload,
        algorithm=algorithm_name,
        environment=environment_name,
        state_size=state_size,
        action_size=action_size,
    )

    agent = build_agent(algorithm_name, state_size, action_size, cfg)
    restore_model_weights(agent, payload)
    agent.qnetwork_local.eval()
    env.close()

    print(f"成功載入模型權重：{model_path}")
    print(f"Algorithm: {algorithm_name}; Environment: {environment_name}")
    print("Evaluation policy: deterministic greedy action selection")
    print("-" * 30)

    # Preserve the previous CLI seed sequence: cfg.SEED + 1, cfg.SEED + 2,
    # ... while reusing the in-memory evaluation helper used by training.
    records, summary = evaluate_policy(
        agent,
        cfg,
        episodes=episodes,
        seed_base=cfg.SEED + 1,
        run_id=model_path.stem,
        render=render,
    )
    for record in records:
        print(f"Test Episode {record['evaluation_episode']}\tScore: {record['return']:.2f}")
    print(f"Mean score across {episodes} episodes: {summary['eval_return_mean']:.2f}")
    return [record["return"] for record in records]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="評估 DQN / Rainbow DQN checkpoint")
    parser.add_argument("--model", required=True, help="checkpoint .pt 或 legacy .pth 路徑")
    parser.add_argument(
        "--algo",
        choices=AGENT_MAPPING.keys(),
        help="只在載入舊式純權重 checkpoint 時必填；metadata checkpoint 會自動辨識。",
    )
    parser.add_argument(
        "--env",
        choices=["Highway", "LunarLander"],
        help="只在載入舊式純權重 checkpoint 時必填；metadata checkpoint 會自動辨識。",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.algo, args.env, args.episodes, render=not args.no_render)

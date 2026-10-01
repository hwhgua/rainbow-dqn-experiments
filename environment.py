"""Gymnasium environment factory and serializable environment snapshots."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


HIGHWAY_ENV_CONFIG = {
    "action": {"type": "DiscreteMetaAction"},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
    },
    "duration": 40,
    "collision_reward": -10,
    "reward_speed_range": [20, 30],
}


def get_environment_snapshot(environment_name):
    """Return the environment settings that should travel with a checkpoint."""
    if environment_name == "Highway":
        return {
            "gym_id": "highway-v0",
            "configuration": deepcopy(HIGHWAY_ENV_CONFIG),
            "wrapper": "FlattenObservation",
        }
    if environment_name == "LunarLander":
        return {
            "gym_id": "LunarLander-v3",
            "configuration": {},
            "wrapper": None,
        }
    raise ValueError(f"不支援的環境：{environment_name}")


def make_environment(environment_name: str, render_mode: str | None = None):
    """Create an environment and apply its project-specific observation setup."""
    if environment_name == "Highway":
        import highway_env  # noqa: F401  # Registers highway-v0.

        env = gym.make("highway-v0", render_mode=render_mode)
        env.unwrapped.configure(deepcopy(HIGHWAY_ENV_CONFIG))
        return FlattenObservation(env)

    if environment_name == "LunarLander":
        return gym.make("LunarLander-v3", render_mode=render_mode)

    raise ValueError(f"不支援的環境：{environment_name}")

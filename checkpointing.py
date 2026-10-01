"""Safe, metadata-rich model checkpoints for training and evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from environment import get_environment_snapshot


CHECKPOINT_FORMAT_VERSION = 1
PROJECT_NAME = "rainbow-dqn-experiments"


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit():
    """Return the current revision when the project is inside a Git repository."""
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


def _stable_hash(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_model_checkpoint(
    agent,
    config,
    state_size,
    action_size,
    *,
    episode,
    average_score,
    global_step,
    training_state_extra=None,
):
    """Build a portable evaluation checkpoint using only safe value types."""
    environment_snapshot = get_environment_snapshot(config.ENV_NAME)
    metadata = {
        "algorithm": config.ALGO_NAME,
        "environment": config.ENV_NAME,
        "state_size": int(state_size),
        "action_size": int(action_size),
        "network_class": type(agent.qnetwork_local).__name__,
        "config": config.to_dict(),
        "environment_config": environment_snapshot,
        "environment_config_sha256": _stable_hash(environment_snapshot),
        "seed": int(config.SEED),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gymnasium": _package_version("gymnasium"),
            "highway_env": _package_version("highway-env"),
        },
        "git_commit": _git_commit(),
    }
    training_state = {
        "episode": int(episode),
        "average_score": float(average_score),
        "global_step": int(global_step),
    }
    if training_state_extra:
        training_state.update(training_state_extra)

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "model",
        "project": PROJECT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": agent.qnetwork_local.state_dict(),
        "training_state": training_state,
        "metadata": metadata,
    }


def save_model_checkpoint(
    path,
    agent,
    config,
    state_size,
    action_size,
    *,
    episode,
    average_score,
    global_step,
    training_state_extra=None,
):
    """Atomically save a portable model checkpoint and return its payload."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_model_checkpoint(
        agent,
        config,
        state_size,
        action_size,
        episode=episode,
        average_score=average_score,
        global_step=global_step,
        training_state_extra=training_state_extra,
    )
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    return payload


def _looks_like_plain_state_dict(payload):
    return bool(payload) and isinstance(payload, dict) and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    )


def load_checkpoint(path, map_location="cpu"):
    """Load a metadata checkpoint or a legacy plain state_dict safely.

    Legacy checkpoints may be used for evaluation or warm starts only because
    they do not identify their originating configuration.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and payload.get("format_version") == CHECKPOINT_FORMAT_VERSION:
        if "model_state_dict" not in payload or "metadata" not in payload:
            raise ValueError("checkpoint 缺少必要的 model_state_dict 或 metadata")
        return payload
    if _looks_like_plain_state_dict(payload):
        return {
            "format_version": 0,
            "checkpoint_kind": "legacy_state_dict",
            "model_state_dict": payload,
            "metadata": None,
        }
    raise ValueError("不支援或不安全的 checkpoint 格式")


def validate_checkpoint_metadata(payload, *, algorithm, environment, state_size, action_size):
    """Reject metadata checkpoints that do not match the runtime configuration."""
    metadata = payload.get("metadata")
    if metadata is None:
        return

    mismatches = []
    expected = {
        "algorithm": algorithm,
        "environment": environment,
        "state_size": int(state_size),
        "action_size": int(action_size),
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            mismatches.append(f"{key}: checkpoint={metadata.get(key)!r}, runtime={expected_value!r}")

    current_environment_hash = _stable_hash(get_environment_snapshot(environment))
    if metadata.get("environment_config_sha256") != current_environment_hash:
        mismatches.append("environment configuration hash differs from the checkpoint")

    if mismatches:
        raise ValueError("checkpoint 與目前設定不相容：\n- " + "\n- ".join(mismatches))


def restore_model_weights(agent, payload):
    """Restore online weights and synchronize the target network when present."""
    agent.qnetwork_local.load_state_dict(payload["model_state_dict"], strict=True)
    target_state_dict = payload.get("target_model_state_dict", payload["model_state_dict"])
    agent.qnetwork_target.load_state_dict(target_state_dict, strict=True)

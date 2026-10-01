"""Training configuration and output-path helpers.

``Config`` intentionally has no file-system side effects. Training creates a
run directory explicitly; evaluation can therefore reconstruct a configuration
without leaving empty ``runs/`` directories behind.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_ALGORITHMS = (
    "VanillaDQN",
    "DDQN",
    "DuelingDQN",
    "PER",
    "NStepPER",
    "NoisyNet",
    "C51",
    "Rainbow",
)


class Config:
    """Settings for one training or evaluation run.

    Highway and LunarLander use different observation scales, network widths,
    and C51 supports. The public showcase defaults to Highway + Rainbow.
    """

    def __init__(self, environment_name="Highway", algorithm_name="Rainbow"):
        if environment_name not in {"Highway", "LunarLander"}:
            raise ValueError("environment_name 必須是 'Highway' 或 'LunarLander'")
        if algorithm_name not in SUPPORTED_ALGORITHMS:
            choices = ", ".join(SUPPORTED_ALGORITHMS)
            raise ValueError(f"不支援的演算法 '{algorithm_name}'。可選：{choices}")

        self.ALGO_NAME = algorithm_name
        self.ENV_NAME = environment_name

        # This is a warm start from a model checkpoint, not a bit-for-bit
        # training resume: replay-buffer and RNG state are intentionally not
        # stored in the portable model checkpoint.
        self.RESUME_TRAIN = False
        self.RESUME_PATH = None
        self.START_EPISODE = 1
        self.N_EPISODES = 2000

        # Shared hyperparameters.
        self.SEED = 42
        self.BUFFER_SIZE = int(1e5)
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.TAU = 1e-3
        self.LR = 5e-4
        self.N_STEP = 3
        self.N_ATOMS = 51

        # Exploration schedules.  The default (None) preserves the original
        # episode-based behavior for ad-hoc training.  The experiment runner
        # sets these to a fixed number of environment steps so every agent is
        # compared on the same schedule.
        self.EPS_ANNEAL_STEPS = None
        self.PER_BETA_START = 0.4
        self.PER_BETA_END = 1.0
        self.PER_BETA_ANNEAL_STEPS = None

        if self.ENV_NAME == "Highway":
            self.UPDATE_EVERY = 4
            self.FC1_UNITS = 256
            self.FC2_UNITS = 256
            self.V_MIN = -50.0
            self.V_MAX = 50.0
            self.EARLY_STOP_SCORE = 25.0
        else:
            self.UPDATE_EVERY = 4
            self.FC1_UNITS = 64
            self.FC2_UNITS = 64
            self.V_MIN = -200.0
            self.V_MAX = 300.0
            self.EARLY_STOP_SCORE = 250.0

        self.MAX_T = 1000
        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.EPS_DECAY = 0.995
        self.RENDER = False

        # These fields are populated by the training entry point and are
        # included in every run/checkpoint snapshot for experiment auditing.
        self.TOTAL_ENV_STEPS = None
        self.EVAL_INTERVAL_STEPS = None
        self.EVAL_EPISODES = None
        self.EVAL_SEED_BASE = None
        self.ENABLE_EARLY_STOP = True
        self.EXPERIMENT_NAME = None

    def to_dict(self):
        """Return a JSON-safe snapshot for run metadata and checkpoints."""
        return dict(vars(self))

    @classmethod
    def from_snapshot(cls, snapshot):
        """Recreate a configuration saved in a metadata checkpoint."""
        if not isinstance(snapshot, dict):
            raise ValueError("checkpoint 的 config snapshot 格式不正確")

        config = cls(
            environment_name=snapshot.get("ENV_NAME", "Highway"),
            algorithm_name=snapshot.get("ALGO_NAME", "Rainbow"),
        )
        allowed_keys = set(config.to_dict())
        for key, value in snapshot.items():
            if key in allowed_keys:
                setattr(config, key, value)
        return config


@dataclass(frozen=True)
class RunPaths:
    """Paths produced only when a new training run starts."""

    run_dir: Path
    best_model_path: Path
    final_model_path: Path
    log_path: Path
    evaluation_episode_log_path: Path
    evaluation_summary_log_path: Path
    summary_path: Path
    config_path: Path


def create_run_paths(config, output_root="runs", run_id=None):
    """Create an output directory for one training run.

    ``run_id`` is intentionally optional.  Interactive training retains the
    timestamped layout, while the experiment runner supplies a stable relative
    identifier such as ``Rainbow/seed_42`` for machine-readable aggregation.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_root = Path(output_root)
    if run_id is None:
        run_dir = output_root / config.ENV_NAME / f"{config.ALGO_NAME}_{timestamp}"
    else:
        run_id_path = Path(run_id)
        if run_id_path.is_absolute() or ".." in run_id_path.parts:
            raise ValueError("run_id 必須是 output_root 內的相對路徑")
        run_dir = output_root / run_id_path
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        best_model_path=run_dir / "best_model.pt",
        final_model_path=run_dir / "final_model.pt",
        log_path=run_dir / "training_log.csv",
        evaluation_episode_log_path=run_dir / "evaluation_episodes.csv",
        evaluation_summary_log_path=run_dir / "evaluation_summary.csv",
        summary_path=run_dir / "run_summary.json",
        config_path=run_dir / "config.json",
    )


def write_run_config(run_paths, config, environment_snapshot):
    """Write a human-readable copy of the settings used for a run."""
    payload = {
        "config": config.to_dict(),
        "environment": environment_snapshot,
        "run_paths": {key: str(value) for key, value in asdict(run_paths).items()},
    }
    run_paths.config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

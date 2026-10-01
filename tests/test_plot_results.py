import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from plot_results import (
    _complete_episodes,
    _trailing_episode_mean,
    plot_training_convergence_overlay_only,
)


def _training_record(run_id, algorithm, seed, step, score, budget_cutoff=False):
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "train_seed": seed,
        "global_step_end": step,
        "score": score,
        "avg_score": score,
        "budget_cutoff": budget_cutoff,
    }


class PlotResultsTest(unittest.TestCase):
    def _write_training_log(self, root, algorithm, seed, rows):
        path = Path(root) / algorithm / f"seed_{seed}"
        path.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(path / "training_log.csv", index=False)

    def test_budget_cutoff_is_excluded_before_trailing_mean(self):
        frame = pd.DataFrame(
            [
                _training_record("Rainbow/seed_0", "Rainbow", 0, 10, 10.0),
                _training_record("Rainbow/seed_0", "Rainbow", 0, 20, 20.0),
                _training_record(
                    "Rainbow/seed_0", "Rainbow", 0, 25, 0.0, budget_cutoff=True
                ),
            ]
        )
        complete = _complete_episodes(frame)

        self.assertEqual(complete["global_step_end"].tolist(), [10, 20])
        np.testing.assert_allclose(_trailing_episode_mean(complete["score"]), [10.0, 15.0])

    def test_overlay_figure_is_created_from_training_logs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_training_log(
                root,
                "VanillaDQN",
                0,
                [
                    _training_record("VanillaDQN/seed_0", "VanillaDQN", 0, 10, 2.0),
                    _training_record("VanillaDQN/seed_0", "VanillaDQN", 0, 20, 4.0),
                ],
            )
            self._write_training_log(
                root,
                "Rainbow",
                0,
                [
                    _training_record("Rainbow/seed_0", "Rainbow", 0, 10, 3.0),
                    _training_record("Rainbow/seed_0", "Rainbow", 0, 20, 6.0),
                ],
            )

            output = plot_training_convergence_overlay_only(root)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

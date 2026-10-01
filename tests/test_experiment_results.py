import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment_results import (
    ExperimentDataError,
    aggregate_evaluations,
    load_evaluation_summaries,
    summarize_run_auc,
)


def _record(run_id, algorithm, seed, step, score):
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "environment": "Highway",
        "train_seed": seed,
        "scheduled_env_step": step,
        "actual_env_step": step,
        "evaluation_round": step // 10,
        "eval_episodes": 2,
        "eval_seed_base": 1000,
        "eval_return_mean": score,
        "eval_return_std": 0.5,
    }


class ExperimentResultsTest(unittest.TestCase):
    def _write_summary(self, root, algorithm, seed, rows):
        path = Path(root) / algorithm / f"seed_{seed}"
        path.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(path / "evaluation_summary.csv", index=False)

    def test_aggregate_uses_training_seed_means(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._write_summary(
                temporary_directory,
                "VanillaDQN",
                0,
                [
                    _record("VanillaDQN/seed_0", "VanillaDQN", 0, 10, 1.0),
                    _record("VanillaDQN/seed_0", "VanillaDQN", 0, 20, 3.0),
                ],
            )
            self._write_summary(
                temporary_directory,
                "VanillaDQN",
                1,
                [
                    _record("VanillaDQN/seed_1", "VanillaDQN", 1, 10, 3.0),
                    _record("VanillaDQN/seed_1", "VanillaDQN", 1, 20, 5.0),
                ],
            )
            self._write_summary(
                temporary_directory,
                "Rainbow",
                0,
                [
                    _record("Rainbow/seed_0", "Rainbow", 0, 10, 10.0),
                    _record("Rainbow/seed_0", "Rainbow", 0, 20, 20.0),
                ],
            )
            self._write_summary(
                temporary_directory,
                "Rainbow",
                1,
                [
                    _record("Rainbow/seed_1", "Rainbow", 1, 10, 14.0),
                    _record("Rainbow/seed_1", "Rainbow", 1, 20, 24.0),
                ],
            )

            summaries = load_evaluation_summaries(temporary_directory)
            aggregate = aggregate_evaluations(summaries)
            vanilla_at_ten = aggregate.loc[
                (aggregate["algorithm"] == "VanillaDQN")
                & (aggregate["scheduled_env_step"] == 10)
            ].iloc[0]
            self.assertEqual(vanilla_at_ten["n_seeds"], 2)
            self.assertAlmostEqual(vanilla_at_ten["mean_eval_return"], 2.0)
            self.assertAlmostEqual(vanilla_at_ten["std_across_seed_means"], 2**0.5)

            auc = summarize_run_auc(summaries)
            self.assertEqual(len(auc), 4)
            self.assertTrue(auc["normalized_eval_auc"].notna().all())

    def test_missing_evaluation_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._write_summary(
                temporary_directory,
                "VanillaDQN",
                0,
                [
                    _record("VanillaDQN/seed_0", "VanillaDQN", 0, 10, 1.0),
                    _record("VanillaDQN/seed_0", "VanillaDQN", 0, 20, 3.0),
                ],
            )
            self._write_summary(
                temporary_directory,
                "VanillaDQN",
                1,
                [_record("VanillaDQN/seed_1", "VanillaDQN", 1, 10, 3.0)],
            )

            with self.assertRaises(ExperimentDataError):
                load_evaluation_summaries(temporary_directory)


if __name__ == "__main__":
    unittest.main()

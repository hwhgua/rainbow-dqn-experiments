import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.per_agent import PERAgent
from config import Config, create_run_paths
from run_experiments import _validate_experiment_spec
from train import _epsilon_for_step


class ExperimentScheduleTest(unittest.TestCase):
    def test_epsilon_schedule_uses_global_steps(self):
        config = Config()
        config.EPS_ANNEAL_STEPS = 100

        self.assertAlmostEqual(_epsilon_for_step(config, 0, 0.7), config.EPS_START)
        self.assertAlmostEqual(_epsilon_for_step(config, 50, 0.7), 0.505)
        self.assertAlmostEqual(_epsilon_for_step(config, 100, 0.7), config.EPS_END)

    def test_per_beta_schedule_uses_global_steps(self):
        agent = SimpleNamespace(
            beta_anneal_steps=100,
            beta_step=49,
            beta_start=0.4,
            beta_end=1.0,
            beta=0.4,
            beta_increment=0.01,
        )
        PERAgent._advance_beta(agent, episode_end=False)
        self.assertEqual(agent.beta_step, 50)
        self.assertAlmostEqual(agent.beta, 0.7)

    def test_experiment_run_id_stays_inside_output_root(self):
        config = Config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_paths = create_run_paths(
                config,
                output_root=temporary_directory,
                run_id="Rainbow/seed_42",
            )
            self.assertEqual(
                run_paths.run_dir,
                Path(temporary_directory) / "Rainbow" / "seed_42",
            )
            with self.assertRaises(ValueError):
                create_run_paths(config, output_root=temporary_directory, run_id="../outside")

    def test_experiment_spec_requires_multiple_evaluation_points(self):
        with self.assertRaises(ValueError):
            _validate_experiment_spec("demo", ["Rainbow"], [0], 100, 100, 2)


if __name__ == "__main__":
    unittest.main()

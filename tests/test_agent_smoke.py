import unittest

import numpy as np

from agent_factory import build_agent
from config import Config, SUPPORTED_ALGORITHMS


class AgentSmokeTest(unittest.TestCase):
    def test_every_public_agent_constructs_acts_and_updates(self):
        """Catch integration errors across all advertised algorithm variants."""
        for algorithm in SUPPORTED_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                config = Config("Highway", algorithm)
                config.BUFFER_SIZE = 64
                config.BATCH_SIZE = 4
                config.UPDATE_EVERY = 1
                config.FC1_UNITS = 8
                config.FC2_UNITS = 8
                config.N_ATOMS = 11
                config.PER_BETA_ANNEAL_STEPS = 12

                agent = build_agent(algorithm, state_size=25, action_size=5, config=config)
                state = np.zeros(25, dtype=np.float32)
                action = agent.act(state, eps=0.0)
                self.assertIn(action, range(5))

                for index in range(12):
                    next_state = np.full(25, index + 1, dtype=np.float32)
                    agent.step(
                        state,
                        action,
                        reward=1.0,
                        next_state=next_state,
                        done=False,
                        episode_end=index == 11,
                    )
                    state = next_state

                self.assertGreater(agent.gradient_updates, 0)


if __name__ == "__main__":
    unittest.main()

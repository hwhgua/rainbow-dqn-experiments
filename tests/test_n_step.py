import unittest

from buffer import NStepDeque


class NStepDequeTest(unittest.TestCase):
    def test_truncated_episode_flushes_without_marking_bellman_terminal(self):
        queue = NStepDeque(n_step=3, gamma=0.9)
        emitted = []
        emitted.extend(queue.append([0], 0, 1.0, [1], False))
        emitted.extend(queue.append([1], 0, 1.0, [2], False))
        emitted.extend(queue.append([2], 0, 1.0, [3], False, episode_end=True))

        self.assertEqual([transition[5] for transition in emitted], [3, 2, 1])
        self.assertEqual([transition[4] for transition in emitted], [False, False, False])
        self.assertEqual([round(transition[2], 2) for transition in emitted], [2.71, 1.9, 1.0])

    def test_terminated_episode_flushes_and_marks_bellman_terminal(self):
        queue = NStepDeque(n_step=3, gamma=0.9)
        emitted = []
        emitted.extend(queue.append([0], 0, 1.0, [1], False))
        emitted.extend(queue.append([1], 0, 2.0, [2], True, episode_end=True))

        self.assertEqual([transition[5] for transition in emitted], [2, 1])
        self.assertEqual([transition[4] for transition in emitted], [True, True])
        self.assertEqual([round(transition[2], 2) for transition in emitted], [2.8, 2.0])


if __name__ == "__main__":
    unittest.main()

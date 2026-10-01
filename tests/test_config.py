import unittest

from config import Config


class ConfigTest(unittest.TestCase):
    def test_public_default_is_highway_rainbow_without_run_directory_side_effect(self):
        config = Config()

        self.assertEqual(config.ENV_NAME, "Highway")
        self.assertEqual(config.ALGO_NAME, "Rainbow")
        self.assertFalse(hasattr(config, "RUN_DIR"))


if __name__ == "__main__":
    unittest.main()

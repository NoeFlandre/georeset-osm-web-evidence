import unittest

from georeset_osm_web_evidence.pipeline.env import (
    get_env_flag,
    get_env_float,
    get_env_int,
)


class PipelineEnvTests(unittest.TestCase):
    def test_reads_typed_environment_values_with_defaults(self) -> None:
        env = {
            "COUNT": "12",
            "DELAY": "1.25",
            "RESET": "1",
            "DISABLED": "0",
        }

        self.assertEqual(get_env_int("COUNT", 5, env=env), 12)
        self.assertEqual(get_env_int("MISSING_COUNT", 5, env=env), 5)
        self.assertEqual(get_env_float("DELAY", 0.5, env=env), 1.25)
        self.assertEqual(get_env_float("MISSING_DELAY", 0.5, env=env), 0.5)
        self.assertEqual(get_env_flag("RESET", env=env), True)
        self.assertEqual(get_env_flag("DISABLED", env=env), False)
        self.assertEqual(get_env_flag("MISSING_FLAG", env=env), False)

    def test_typed_environment_errors_name_the_invalid_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "COUNT"):
            get_env_int("COUNT", 5, env={"COUNT": "many"})

        with self.assertRaisesRegex(ValueError, "DELAY"):
            get_env_float("DELAY", 0.5, env={"DELAY": "slow"})


if __name__ == "__main__":
    unittest.main()

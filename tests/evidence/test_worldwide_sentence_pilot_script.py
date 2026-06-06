import logging
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from scripts.evidence.run_worldwide_sentence_pilot import collect_search_results


class WorldwideSentencePilotScriptTests(unittest.TestCase):
    def _silent_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    def _pilot_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                    "osm_tags": {"name": "Forest A", "landuse": "forest"},
                }
            ]
        )

    def test_collect_search_results_records_request_failures(self) -> None:
        logger = self._silent_logger("test_collect_search_results_request_failure")

        with patch(
            "scripts.evidence.run_worldwide_sentence_pilot.build_pilot_queries",
            return_value=[("en", '"Forest A" forest')],
        ), patch(
            "scripts.evidence.run_worldwide_sentence_pilot.search_brave",
            side_effect=requests.Timeout("search timed out"),
        ), patch("scripts.evidence.run_worldwide_sentence_pilot.time.sleep"):
            results_df, attempts_df = collect_search_results(
                self._pilot_dataframe(),
                logger,
            )

        self.assertTrue(results_df.empty)
        self.assertEqual(len(attempts_df), 1)
        self.assertEqual(attempts_df.loc[0, "result_count"], 0)
        self.assertIn("search timed out", attempts_df.loc[0, "search_error"])

    def test_collect_search_results_propagates_configuration_errors(self) -> None:
        logger = self._silent_logger("test_collect_search_results_configuration_error")

        with patch(
            "scripts.evidence.run_worldwide_sentence_pilot.build_pilot_queries",
            return_value=[("en", '"Forest A" forest')],
        ), patch(
            "scripts.evidence.run_worldwide_sentence_pilot.search_brave",
            side_effect=ValueError("BRAVE_SEARCH_API_KEY is not set"),
        ), patch("scripts.evidence.run_worldwide_sentence_pilot.time.sleep"):
            with self.assertRaisesRegex(ValueError, "BRAVE_SEARCH_API_KEY"):
                collect_search_results(self._pilot_dataframe(), logger)


if __name__ == "__main__":
    unittest.main()

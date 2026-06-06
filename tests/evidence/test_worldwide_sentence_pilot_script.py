import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import requests

from scripts.evidence.run_worldwide_sentence_pilot import (
    collect_search_results,
    fetch_candidate_pages,
    load_or_collect_search_results,
    load_or_build_dataframe,
    pilot_artifact_is_usable,
    search_attempts_cover_expected_queries,
)


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

    def test_pilot_artifact_requires_local_language_metadata(self) -> None:
        stale_pilot_df = pd.DataFrame(
            [
                {
                    "polygon_name": "Forest A",
                    "polygon_category": "forest",
                    "has_wikipedia_articles": None,
                }
            ]
        )

        self.assertFalse(pilot_artifact_is_usable(stale_pilot_df))

    def test_search_attempt_coverage_requires_every_expected_query(self) -> None:
        pilot_df = self._pilot_dataframe()
        attempts_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query_language": "en",
                    "query": '"Forest A" forest',
                }
            ]
        )

        with patch(
            "scripts.evidence.run_worldwide_sentence_pilot.build_pilot_queries",
            return_value=[
                ("en", '"Forest A" forest'),
                ("fr", '"Forest A" forêt'),
            ],
        ):
            self.assertFalse(
                search_attempts_cover_expected_queries(attempts_df, pilot_df)
            )

            complete_attempts_df = pd.concat(
                [
                    attempts_df,
                    pd.DataFrame(
                        [
                            {
                                "osm_type": "way",
                                "osm_id": 1,
                                "query_language": "fr",
                                "query": '"Forest A" forêt',
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            self.assertTrue(
                search_attempts_cover_expected_queries(complete_attempts_df, pilot_df)
            )

    def test_load_or_collect_search_results_reports_when_it_rebuilds(self) -> None:
        pilot_df = self._pilot_dataframe()

        with TemporaryDirectory() as temporary_directory:
            search_results_path = Path(temporary_directory) / "search_results.parquet"
            search_attempts_path = Path(temporary_directory) / "search_attempts.parquet"
            pd.DataFrame([{"url": "https://stale.example"}]).to_parquet(
                search_results_path,
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "osm_type": "way",
                        "osm_id": 1,
                        "query_language": "en",
                        "query": '"Forest A" forest',
                    }
                ]
            ).to_parquet(search_attempts_path, index=False)

            rebuilt_results_df = pd.DataFrame([{"url": "https://fresh.example"}])
            rebuilt_attempts_df = pd.DataFrame(
                [
                    {
                        "osm_type": "way",
                        "osm_id": 1,
                        "query_language": "en",
                        "query": '"Forest A" forest',
                    },
                    {
                        "osm_type": "way",
                        "osm_id": 1,
                        "query_language": "fr",
                        "query": '"Forest A" forêt',
                    },
                ]
            )

            with patch(
                "scripts.evidence.run_worldwide_sentence_pilot.SEARCH_RESULTS_PATH",
                search_results_path,
            ), patch(
                "scripts.evidence.run_worldwide_sentence_pilot.SEARCH_ATTEMPTS_PATH",
                search_attempts_path,
            ), patch(
                "scripts.evidence.run_worldwide_sentence_pilot.build_pilot_queries",
                return_value=[
                    ("en", '"Forest A" forest'),
                    ("fr", '"Forest A" forêt'),
                ],
            ), patch(
                "scripts.evidence.run_worldwide_sentence_pilot.collect_search_results",
                return_value=(rebuilt_results_df, rebuilt_attempts_df),
            ) as collect_search_results:
                results_df, attempts_df, rebuilt = load_or_collect_search_results(
                    pilot_df,
                    self._silent_logger("test_search_rebuild_flag"),
                    reset=False,
                )

        collect_search_results.assert_called_once()
        self.assertTrue(rebuilt)
        self.assertEqual(results_df["url"].to_list(), ["https://fresh.example"])
        self.assertEqual(len(attempts_df), 2)

    def test_load_or_build_dataframe_reuses_existing_artifact(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stage.parquet"
            existing_df = pd.DataFrame([{"value": "cached"}])
            existing_df.to_parquet(path, index=False)

            result = load_or_build_dataframe(
                path=path,
                stage_name="test stage",
                logger=self._silent_logger("test_load_existing_stage"),
                build=lambda: pd.DataFrame([{"value": "rebuilt"}]),
                reset=False,
            )

        self.assertEqual(result["value"].to_list(), ["cached"])

    def test_load_or_build_dataframe_rebuilds_when_reset_is_requested(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stage.parquet"
            pd.DataFrame([{"value": "cached"}]).to_parquet(path, index=False)

            result = load_or_build_dataframe(
                path=path,
                stage_name="test stage",
                logger=self._silent_logger("test_reset_stage"),
                build=lambda: pd.DataFrame([{"value": "rebuilt"}]),
                reset=True,
            )
            saved_df = pd.read_parquet(path)

        self.assertEqual(result["value"].to_list(), ["rebuilt"])
        self.assertEqual(saved_df["value"].to_list(), ["rebuilt"])

    def test_fetch_candidate_pages_reuses_existing_rows_and_checkpoints_new_rows(self) -> None:
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": "https://example.org/a",
                    "title": "A",
                    "description": "desc A",
                    "queries": ["q1"],
                    "best_rank": 1,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Forest B",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": "https://example.org/b",
                    "title": "B",
                    "description": "desc B",
                    "queries": ["q2"],
                    "best_rank": 1,
                    "world_region": "Asia",
                    "country": "Sri Lanka",
                    "local_language": "si",
                    "query_local_language": "si",
                    "area_size_bin": "small",
                    "polygon_category": "forest",
                },
            ]
        )
        existing_row = {
            "osm_type": "way",
            "osm_id": 1,
            "polygon_name": "Forest A",
            "has_wikipedia_articles": None,
            "provider": "brave",
            "source_url": "https://example.org/a",
            "search_title": "A",
            "search_description": "desc A",
            "search_queries": "q1",
            "url": "https://example.org/a",
            "final_url": "https://example.org/a",
            "status_code": 200,
            "title": "A",
            "text": "Cached text.",
            "text_length": 12,
            "fetch_error": None,
            "extraction_method": "trafilatura",
            "extraction_error": None,
            "best_rank": 1,
            "world_region": "Europe",
            "country": "France",
            "local_language": "fr",
            "query_local_language": "fr",
            "area_size_bin": "medium",
            "polygon_category": "forest",
        }

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "page_text.parquet"
            pd.DataFrame([existing_row]).to_parquet(output_path, index=False)

            with patch(
                "scripts.evidence.run_worldwide_sentence_pilot.fetch_page_text",
                return_value={
                    "url": "https://example.org/b",
                    "final_url": "https://example.org/b",
                    "status_code": 200,
                    "title": "B",
                    "text": "New text.",
                    "text_length": 9,
                    "fetch_error": None,
                    "extraction_method": "trafilatura",
                    "extraction_error": None,
                },
            ) as fetch_page_text, patch(
                "scripts.evidence.run_worldwide_sentence_pilot.time.sleep"
            ):
                result, changed = fetch_candidate_pages(
                    candidate_urls_df,
                    self._silent_logger("test_fetch_candidate_pages_checkpoint"),
                    output_path=output_path,
                    reset=False,
                )

            saved_df = pd.read_parquet(output_path)

        fetch_page_text.assert_called_once_with("https://example.org/b")
        self.assertTrue(changed)
        self.assertEqual(result["source_url"].to_list(), [
            "https://example.org/a",
            "https://example.org/b",
        ])
        self.assertEqual(saved_df["source_url"].to_list(), [
            "https://example.org/a",
            "https://example.org/b",
        ])


if __name__ == "__main__":
    unittest.main()

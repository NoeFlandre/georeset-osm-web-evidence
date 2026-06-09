import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
import requests

from georeset_osm_web_evidence.evidence.page_text import (
    fetch_candidate_pages,
    page_text_quality_artifact_is_usable,
)
from scripts.evidence.run_worldwide_sentence_pilot import (
    build_pilot_sentence_candidates,
    collect_search_results,
    load_or_collect_search_results,
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

    def test_build_pilot_sentence_candidates_applies_sampling_limits(self) -> None:
        sentence_terms = [
            "oak",
            "pine",
            "birch",
            "cedar",
            "willow",
            "maple",
            "beech",
            "ash",
            "elm",
            "spruce",
            "hemlock",
            "larch",
        ]
        sentence_rows = []
        for url_index, sentence_term in enumerate(sentence_terms):
            for sentence_index in range(2):
                sentence_rows.append(
                    {
                        "osm_type": "way",
                        "osm_id": 1,
                        "url": f"https://example.org/page-{url_index}",
                        "sentence": (
                            f"The {sentence_term} habitat supports forest "
                            f"biodiversity and wetland edges case {sentence_index}."
                        ),
                    }
                )
        raw_sentence_df = pd.DataFrame(sentence_rows)
        pilot_df = self._pilot_dataframe()

        with patch(
            "scripts.evidence.run_worldwide_sentence_pilot.build_sentence_candidate_dataframe",
            return_value=raw_sentence_df,
        ):
            result = build_pilot_sentence_candidates(
                pd.DataFrame([{"text": "unused"}]),
                pilot_df,
            )

        self.assertEqual(len(result), 10)
        self.assertTrue(result.groupby(["osm_type", "osm_id", "url"]).size().eq(1).all())
        self.assertEqual(
            result["sentence"].to_list(),
            [
                (
                    f"The {sentence_term} habitat supports forest biodiversity "
                    "and wetland edges case 0."
                )
                for sentence_term in sentence_terms[:10]
            ],
        )
        self.assertEqual(result["world_region"].to_list(), ["Europe"] * 10)

    def test_build_pilot_sentence_candidates_deduplicates_before_sampling(self) -> None:
        duplicate_sentence = (
            "The protected forest contains wetlands and grassland habitats "
            "near the river valley today."
        )
        near_duplicate_sentence = (
            "The protected forest contains wetlands and grassland habitats "
            "near the river valley today area."
        )
        sentence_rows = [
            {
                "osm_type": "way",
                "osm_id": 1,
                "url": "https://example.org/duplicate-a",
                "sentence": duplicate_sentence,
            },
            {
                "osm_type": "way",
                "osm_id": 1,
                "url": "https://example.org/duplicate-b",
                "sentence": near_duplicate_sentence,
            },
        ]
        sentence_terms = [
            "oak",
            "pine",
            "birch",
            "cedar",
            "willow",
            "maple",
            "beech",
            "ash",
            "elm",
            "spruce",
        ]
        for url_index, sentence_term in enumerate(sentence_terms):
            sentence_rows.append(
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": f"https://example.org/page-{url_index}",
                    "sentence": (
                        f"The {sentence_term} woodland corridor contains "
                        "distinct environmental content for testing."
                    ),
                }
            )
        raw_sentence_df = pd.DataFrame(sentence_rows)

        with patch(
            "scripts.evidence.run_worldwide_sentence_pilot.build_sentence_candidate_dataframe",
            return_value=raw_sentence_df,
        ):
            result = build_pilot_sentence_candidates(
                pd.DataFrame([{"text": "unused"}]),
                self._pilot_dataframe(),
            )

        self.assertEqual(len(result), 10)
        self.assertNotIn(near_duplicate_sentence, result["sentence"].to_list())
        self.assertIn("https://example.org/page-8", result["url"].to_list())

    def test_page_text_quality_artifact_rejects_missing_query_language(self) -> None:
        page_text_with_quality_df = pd.DataFrame(
            [
                {
                    "source_url": "https://example.org/a",
                    "text": "Cached text.",
                    "quality_score": 1.0,
                    "query_language": pd.NA,
                }
            ]
        )

        self.assertFalse(
            page_text_quality_artifact_is_usable(page_text_with_quality_df)
        )

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

            fetch_page_text = Mock(
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
            )
            result, changed = fetch_candidate_pages(
                candidate_urls_df,
                self._silent_logger("test_fetch_candidate_pages_checkpoint"),
                output_path=output_path,
                reset=False,
                fetch_page_text_func=fetch_page_text,
                sleep_func=lambda _: None,
            )

            saved_df = pd.read_parquet(output_path)

        fetch_page_text.assert_called_once_with(
            "https://example.org/b",
            timeout_seconds=10,
        )
        self.assertTrue(changed)
        self.assertEqual(result["source_url"].to_list(), [
            "https://example.org/a",
            "https://example.org/b",
        ])
        self.assertEqual(saved_df["source_url"].to_list(), [
            "https://example.org/a",
            "https://example.org/b",
        ])

    def test_fetch_candidate_pages_backfills_cached_query_language(self) -> None:
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
                    "query_language": "en",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                }
            ]
        )
        cached_row = {
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
        }

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "page_text.parquet"
            pd.DataFrame([cached_row]).to_parquet(output_path, index=False)

            fetch_page_text = Mock()
            result, changed = fetch_candidate_pages(
                candidate_urls_df,
                self._silent_logger("test_fetch_candidate_pages_backfill"),
                output_path=output_path,
                reset=False,
                fetch_page_text_func=fetch_page_text,
                sleep_func=lambda _: None,
            )

            saved_df = pd.read_parquet(output_path)

        fetch_page_text.assert_not_called()
        self.assertTrue(changed)
        self.assertEqual(result.loc[0, "query_language"], "en")
        self.assertEqual(saved_df.loc[0, "query_language"], "en")

    def test_fetch_candidate_pages_prunes_cached_rows_outside_candidates(self) -> None:
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": "https://example.org/keep",
                    "title": "Keep",
                    "description": "Keep desc",
                    "queries": ["q1"],
                    "best_rank": 1,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "query_language": "en",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                }
            ]
        )
        cached_rows = pd.DataFrame(
            [
                {
                    "source_url": "https://example.org/keep",
                    "url": "https://example.org/keep",
                    "text": "Keep text.",
                    "text_length": 10,
                },
                {
                    "source_url": "https://example.org/drop",
                    "url": "https://example.org/drop",
                    "text": "Drop text.",
                    "text_length": 10,
                },
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "page_text.parquet"
            cached_rows.to_parquet(output_path, index=False)

            fetch_page_text = Mock()
            result, changed = fetch_candidate_pages(
                candidate_urls_df,
                self._silent_logger("test_fetch_candidate_pages_prune"),
                output_path=output_path,
                reset=False,
                fetch_page_text_func=fetch_page_text,
                sleep_func=lambda _: None,
            )

            saved_df = pd.read_parquet(output_path)

        fetch_page_text.assert_not_called()
        self.assertTrue(changed)
        self.assertEqual(result["source_url"].to_list(), ["https://example.org/keep"])
        self.assertEqual(saved_df["source_url"].to_list(), result["source_url"].to_list())

    def test_fetch_candidate_pages_stops_when_quota_callback_is_satisfied(self) -> None:
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": f"https://example.org/{index}",
                    "title": f"Title {index}",
                    "description": f"Desc {index}",
                    "queries": [f"q{index}"],
                    "best_rank": index,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                }
                for index in range(3)
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "page_text.parquet"

            def stop_when(page_text_df: pd.DataFrame) -> bool:
                return len(page_text_df) >= 2

            fetch_page_text = Mock(
                side_effect=lambda url, timeout_seconds: {
                    "url": url,
                    "final_url": url,
                    "status_code": 200,
                    "title": "Fetched",
                    "text": "Fetched text.",
                    "text_length": 13,
                    "fetch_error": None,
                    "extraction_method": "trafilatura",
                    "extraction_error": None,
                },
            )
            result, changed = fetch_candidate_pages(
                candidate_urls_df,
                self._silent_logger("test_fetch_candidate_pages_early_stop"),
                output_path=output_path,
                reset=False,
                stop_when=stop_when,
                stop_check_interval=1,
                fetch_page_text_func=fetch_page_text,
                sleep_func=lambda _: None,
            )

        self.assertTrue(changed)
        self.assertEqual(len(result), 2)
        self.assertEqual(fetch_page_text.call_count, 2)

    def test_fetch_candidate_pages_checkpoints_pdf_without_fetching_it(self) -> None:
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": "https://example.org/report.pdf",
                    "title": "PDF",
                    "description": "PDF desc",
                    "queries": ["q-pdf"],
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
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": "https://example.org/page",
                    "title": "HTML",
                    "description": "HTML desc",
                    "queries": ["q-html"],
                    "best_rank": 2,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                },
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "page_text.parquet"

            fetch_page_text = Mock(
                return_value={
                    "url": "https://example.org/page",
                    "final_url": "https://example.org/page",
                    "status_code": 200,
                    "title": "HTML",
                    "text": "New text.",
                    "text_length": 9,
                    "fetch_error": None,
                    "extraction_method": "trafilatura",
                    "extraction_error": None,
                },
            )
            result, changed = fetch_candidate_pages(
                candidate_urls_df,
                self._silent_logger("test_fetch_candidate_pages_pdf_skip"),
                output_path=output_path,
                reset=False,
                fetch_page_text_func=fetch_page_text,
                sleep_func=lambda _: None,
            )

            saved_df = pd.read_parquet(output_path)

        fetch_page_text.assert_called_once_with(
            "https://example.org/page",
            timeout_seconds=10,
        )
        self.assertTrue(changed)
        self.assertEqual(result["source_url"].to_list(), [
            "https://example.org/report.pdf",
            "https://example.org/page",
        ])
        self.assertEqual(saved_df["source_url"].to_list(), result["source_url"].to_list())
        pdf_row = result[result["source_url"] == "https://example.org/report.pdf"].iloc[0]
        self.assertEqual(pdf_row["text_length"], 0)
        self.assertEqual(pdf_row["fetch_error"], "Skipped PDF URL")


if __name__ == "__main__":
    unittest.main()

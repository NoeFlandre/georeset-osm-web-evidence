import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import georeset_osm_web_evidence.search.coverage as coverage
from georeset_osm_web_evidence.search.coverage import (
    build_expected_query_table,
    choose_unsearched_polygons,
    find_missing_queries,
    load_existing_search_attempts,
    load_existing_search_results,
    summarize_search_coverage,
)


class SearchCoverageTests(unittest.TestCase):
    def test_summarizes_searched_and_unsearched_polygons(self):
        polygons = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1},
                {"osm_type": "way", "osm_id": 2},
                {"osm_type": "relation", "osm_id": 3},
            ]
        )
        results = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "query": '"A" forêt'},
                {"osm_type": "way", "osm_id": 1, "query": '"A" biodiversité'},
            ]
        )

        summary = summarize_search_coverage(polygons, results)

        self.assertEqual(summary["total_polygons"], 3)
        self.assertEqual(summary["searched_polygons"], 1)
        self.assertEqual(summary["unsearched_polygons"], 2)
        self.assertEqual(summary["searched_queries"], 2)

    def test_choose_unsearched_polygons_balances_wikipedia_status(self):
        polygons = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "has_wikipedia_articles": True},
                {"osm_type": "way", "osm_id": 2, "has_wikipedia_articles": True},
                {"osm_type": "way", "osm_id": 3, "has_wikipedia_articles": False},
                {"osm_type": "way", "osm_id": 4, "has_wikipedia_articles": False},
            ]
        )
        results = pd.DataFrame([{"osm_type": "way", "osm_id": 1}])

        selected = choose_unsearched_polygons(
            polygons,
            results,
            polygon_limit=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["has_wikipedia_articles"]), {True, False})
        self.assertNotIn(1, selected["osm_id"].to_list())

    def test_choose_unsearched_polygons_uses_attempt_log(self):
        polygons = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "has_wikipedia_articles": True},
                {"osm_type": "way", "osm_id": 2, "has_wikipedia_articles": False},
            ]
        )
        results = pd.DataFrame(columns=["osm_type", "osm_id"])
        attempts = pd.DataFrame([{"osm_type": "way", "osm_id": 1}])

        selected = choose_unsearched_polygons(
            polygons,
            results,
            polygon_limit=2,
            attempted_polygons_df=attempts,
        )

        self.assertEqual(selected["osm_id"].to_list(), [2])

    def test_finds_missing_queries_from_results_and_attempts(self):
        polygons = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": True,
                    "osm_tags": {"name": "Forest A", "landuse": "forest"},
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Marsh B",
                    "has_wikipedia_articles": False,
                    "osm_tags": {"name": "Marsh B", "natural": "wetland"},
                },
            ]
        )
        results = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "query": '"Forest A" forêt'},
            ]
        )
        attempts = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 2, "query": '"Marsh B" zone humide'},
            ]
        )

        expected = build_expected_query_table(polygons)
        missing = find_missing_queries(expected, results, attempts)

        self.assertEqual(len(expected), 8)
        self.assertEqual(len(missing), 6)
        self.assertNotIn('"Forest A" forêt', missing["query"].to_list())
        self.assertNotIn('"Marsh B" zone humide', missing["query"].to_list())

    def test_builds_expected_queries_for_explicit_search_languages(self):
        polygons = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": True,
                    "osm_tags": {"name": "Forest A", "landuse": "forest"},
                },
            ]
        )

        expected = build_expected_query_table(polygons, search_languages=["fr", "en"])

        self.assertEqual(
            expected["query"].to_list(),
            [
                '"Forest A" forêt',
                '"Forest A" biodiversité',
                '"Forest A" gestion forestière',
                '"Forest A" environnement',
                '"Forest A" forest',
                '"Forest A" biodiversity',
                '"Forest A" forest management',
                '"Forest A" environment',
            ],
        )

    def test_load_existing_search_results_returns_schema_for_missing_file(self):
        with TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.parquet"

            result = load_existing_search_results(missing_path)

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.to_list(), ["osm_type", "osm_id", "query"])

    def test_load_existing_search_attempts_rejects_file_without_query_column(self):
        with TemporaryDirectory() as temporary_directory:
            attempts_path = Path(temporary_directory) / "attempts.parquet"
            pd.DataFrame([{"osm_type": "way", "osm_id": 1}]).to_parquet(
                attempts_path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "query"):
                load_existing_search_attempts(attempts_path)

    def test_empty_expected_query_table_keeps_mergeable_schema(self):
        polygons = pd.DataFrame(
            columns=[
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "osm_tags",
            ]
        )
        results = pd.DataFrame(columns=["osm_type", "osm_id", "query"])

        expected = build_expected_query_table(polygons)
        missing = find_missing_queries(expected, results)

        self.assertTrue(expected.empty)
        self.assertEqual(
            expected.columns.to_list(),
            [
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "query",
            ],
        )
        self.assertTrue(missing.empty)

    def test_choose_unsearched_polygons_respects_zero_limit(self):
        polygons = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "has_wikipedia_articles": True},
                {"osm_type": "way", "osm_id": 2, "has_wikipedia_articles": False},
            ]
        )
        results = pd.DataFrame(columns=["osm_type", "osm_id"])

        selected = choose_unsearched_polygons(
            polygons,
            results,
            polygon_limit=0,
        )

        self.assertTrue(selected.empty)

    def test_choose_polygons_to_search_can_complete_existing_polygons_only(self):
        polygons = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "has_wikipedia_articles": True},
                {"osm_type": "way", "osm_id": 2, "has_wikipedia_articles": False},
                {"osm_type": "relation", "osm_id": 3, "has_wikipedia_articles": True},
            ]
        )
        existing_results = pd.DataFrame([{"osm_type": "way", "osm_id": 1}])
        existing_attempts = pd.DataFrame([{"osm_type": "relation", "osm_id": 3}])

        self.assertTrue(hasattr(coverage, "choose_polygons_to_search"))
        selected = coverage.choose_polygons_to_search(
            polygons,
            existing_results,
            existing_attempts,
            new_polygon_limit=10,
            complete_existing_polygons_only=True,
        )

        self.assertEqual(
            selected[["osm_type", "osm_id"]].to_dict("records"),
            [
                {"osm_type": "way", "osm_id": 1},
                {"osm_type": "relation", "osm_id": 3},
            ],
        )

    def test_choose_polygons_to_search_selects_unsearched_polygons_when_not_completing(self):
        polygons = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "has_wikipedia_articles": True},
                {"osm_type": "way", "osm_id": 2, "has_wikipedia_articles": False},
                {"osm_type": "way", "osm_id": 3, "has_wikipedia_articles": True},
            ]
        )
        existing_results = pd.DataFrame([{"osm_type": "way", "osm_id": 1}])
        existing_attempts = pd.DataFrame([{"osm_type": "way", "osm_id": 2}])

        self.assertTrue(hasattr(coverage, "choose_polygons_to_search"))
        selected = coverage.choose_polygons_to_search(
            polygons,
            existing_results,
            existing_attempts,
            new_polygon_limit=2,
            complete_existing_polygons_only=False,
        )

        self.assertEqual(selected["osm_id"].to_list(), [3])


if __name__ == "__main__":
    unittest.main()

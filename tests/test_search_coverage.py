import unittest

import pandas as pd

from georeset_osm_web_evidence.search.coverage import (
    build_expected_query_table,
    choose_unsearched_polygons,
    find_missing_queries,
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


if __name__ == "__main__":
    unittest.main()

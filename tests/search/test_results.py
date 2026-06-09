import unittest
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

import georeset_osm_web_evidence.search.results as search_results
from georeset_osm_web_evidence.search.results import (
    attempt_to_row,
    result_to_row,
)


class SearchResultsTests(unittest.TestCase):
    def test_builds_search_result_row(self):
        polygon_row = SimpleNamespace(
            osm_type="way",
            osm_id=123,
            has_wikipedia_articles=True,
        )
        result = {
            "provider": "brave",
            "title": "Result title",
            "url": "https://example.com/result",
            "description": "Result description",
        }

        row = result_to_row(
            polygon_row=polygon_row,
            polygon_name="Forêt Alpha",
            query='"Forêt Alpha" forest',
            rank=2,
            result=result,
        )

        self.assertEqual(
            row,
            {
                "osm_type": "way",
                "osm_id": 123,
                "polygon_name": "Forêt Alpha",
                "has_wikipedia_articles": True,
                "query": '"Forêt Alpha" forest',
                "provider": "brave",
                "rank": 2,
                "title": "Result title",
                "url": "https://example.com/result",
                "description": "Result description",
            },
        )

    def test_builds_search_attempt_row(self):
        polygon_row = SimpleNamespace(
            osm_type="relation",
            osm_id=456,
            has_wikipedia_articles=False,
        )

        row = attempt_to_row(
            polygon_row=polygon_row,
            polygon_name="Marais Beta",
            query='"Marais Beta" wetland',
            result_count=5,
        )

        self.assertEqual(row["osm_type"], "relation")
        self.assertEqual(row["osm_id"], 456)
        self.assertEqual(row["polygon_name"], "Marais Beta")
        self.assertEqual(row["has_wikipedia_articles"], False)
        self.assertEqual(row["query"], '"Marais Beta" wetland')
        self.assertEqual(row["result_count"], 5)

        attempted_at = datetime.fromisoformat(row["attempted_at"])
        self.assertIsNotNone(attempted_at.tzinfo)

    def test_prepares_candidate_urls_without_wikipedia_and_with_query_provenance(self):
        search_results_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Bois Alpha",
                    "has_wikipedia_articles": False,
                    "provider": "brave",
                    "url": "https://example.com/forest",
                    "rank": 3,
                    "title": "Later title",
                    "description": "Later description",
                    "query": '"Bois Alpha" "France" "forest"',
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Bois Alpha",
                    "has_wikipedia_articles": False,
                    "provider": "brave",
                    "url": "https://example.com/forest",
                    "rank": 1,
                    "title": "Best title",
                    "description": "Best description",
                    "query": '"Bois Alpha" "France" "wood"',
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Bois Alpha",
                    "has_wikipedia_articles": False,
                    "provider": "brave",
                    "url": "https://en.wikipedia.org/wiki/Bois_Alpha",
                    "rank": 2,
                    "title": "Wikipedia",
                    "description": "Filtered",
                    "query": '"Bois Alpha" "France" "forest"',
                },
            ]
        )

        self.assertTrue(hasattr(search_results, "prepare_candidate_urls"))

        candidate_urls_df = search_results.prepare_candidate_urls(search_results_df)

        self.assertEqual(len(candidate_urls_df), 1)
        self.assertEqual(candidate_urls_df.loc[0, "url"], "https://example.com/forest")
        self.assertEqual(candidate_urls_df.loc[0, "best_rank"], 1)
        self.assertEqual(candidate_urls_df.loc[0, "title"], "Best title")
        self.assertEqual(candidate_urls_df.loc[0, "description"], "Best description")
        self.assertEqual(
            candidate_urls_df.loc[0, "queries"],
            [
                '"Bois Alpha" "France" "forest"',
                '"Bois Alpha" "France" "wood"',
            ],
        )

    def test_merges_search_results_and_attempts_without_replacing_existing_rows(self):
        existing_results_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query": '"Forest A" forest',
                    "url": "https://example.com/a",
                    "title": "Existing title",
                }
            ]
        )
        new_results_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query": '"Forest A" forest',
                    "url": "https://example.com/a",
                    "title": "New duplicate title",
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query": '"Forest A" forest',
                    "url": "https://example.com/b",
                    "title": "New title",
                },
            ]
        )
        existing_attempts_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query": '"Forest A" forest',
                    "result_count": 1,
                }
            ]
        )
        new_attempts_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query": '"Forest A" forest',
                    "result_count": 2,
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "query": '"Forest A" biodiversity',
                    "result_count": 3,
                },
            ]
        )

        self.assertTrue(hasattr(search_results, "merge_search_results"))
        self.assertTrue(hasattr(search_results, "merge_search_attempts"))
        merged_results_df = search_results.merge_search_results(
            existing_results_df,
            new_results_df,
        )
        merged_attempts_df = search_results.merge_search_attempts(
            existing_attempts_df,
            new_attempts_df,
        )

        self.assertEqual(
            merged_results_df[["url", "title"]].to_dict("records"),
            [
                {"url": "https://example.com/a", "title": "Existing title"},
                {"url": "https://example.com/b", "title": "New title"},
            ],
        )
        self.assertEqual(
            merged_attempts_df[["query", "result_count"]].to_dict("records"),
            [
                {"query": '"Forest A" forest', "result_count": 1},
                {"query": '"Forest A" biodiversity', "result_count": 3},
            ],
        )


if __name__ == "__main__":
    unittest.main()

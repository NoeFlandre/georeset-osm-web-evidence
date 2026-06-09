import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.context_query_search import (
    build_context_query_search_artifacts,
    search_context_queries_for_polygon,
)


class ContextQuerySearchTests(unittest.TestCase):
    def test_empty_search_artifacts_keep_expected_columns(self):
        search_results_df, search_attempts_df = build_context_query_search_artifacts(
            pd.DataFrame(),
            search_func=lambda _query, _count, **_kwargs: [],
            sleep_func=lambda _seconds: None,
        )

        self.assertEqual(
            search_results_df.columns.to_list(),
            [
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "query",
                "provider",
                "rank",
                "title",
                "url",
                "description",
                "query_language",
                "world_region",
                "country",
                "local_language",
                "query_local_language",
                "area_size_bin",
                "polygon_category",
            ],
        )
        self.assertEqual(
            search_attempts_df.columns.to_list(),
            [
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "query",
                "attempted_at",
                "result_count",
                "query_language",
                "search_error",
            ],
        )

    def test_searches_polygon_with_context_queries(self):
        polygon_row = next(
            pd.DataFrame(
                [
                    {
                        "osm_type": "way",
                        "osm_id": 1,
                        "osm_tags": {
                            "name": "Sagole Baobab",
                            "leisure": "nature_reserve",
                        },
                        "polygon_name": "Sagole Baobab",
                        "has_wikipedia_articles": False,
                        "country": "South Africa",
                        "world_region": "Africa",
                        "source_extract_id": "south-africa",
                        "polygon_category": "protected_area",
                        "local_language": "en",
                        "query_local_language": "en",
                        "area_size_bin": "tiny",
                    }
                ]
            ).itertuples()
        )
        seen_queries = []

        def fake_search(query: str, count: int, **kwargs):
            seen_queries.append(query)
            self.assertEqual(count, 1)
            self.assertEqual(kwargs["country"], "US")
            self.assertEqual(kwargs["search_lang"], "en")
            return [
                {
                    "provider": "brave",
                    "title": "Search result",
                    "url": f"https://example.org/{len(seen_queries)}",
                    "description": "Description",
                }
            ]

        search_results_df, search_attempts_df = search_context_queries_for_polygon(
            polygon_row,
            search_func=fake_search,
            sleep_func=lambda _seconds: None,
            max_queries_per_polygon=4,
            results_per_query=1,
            request_delay_seconds=0,
        )

        self.assertEqual(len(seen_queries), 4)
        self.assertTrue(
            all("South Africa" in query or "Africa" in query for query in seen_queries)
        )
        self.assertIn(
            '"Sagole Baobab" "South Africa" protected area',
            seen_queries,
        )
        self.assertEqual(search_results_df["query_language"].unique().tolist(), ["en"])
        self.assertEqual(search_attempts_df["query_language"].unique().tolist(), ["en"])


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.location_topic_search import (
    build_location_topic_search_artifacts,
    search_location_topic_for_polygon,
)


class LocationTopicSearchTests(unittest.TestCase):
    def test_searches_polygon_with_location_topic_queries(self):
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

        search_results_df, search_attempts_df = search_location_topic_for_polygon(
            polygon_row,
            search_func=fake_search,
            sleep_func=lambda _seconds: None,
            max_queries_per_polygon=4,
            results_per_query=1,
            request_delay_seconds=0,
        )

        self.assertEqual(
            seen_queries,
            [
                '"Sagole Baobab" "South Africa" "nature reserve"',
                '"Sagole Baobab" "South Africa" "protection"',
                '"Sagole Baobab" "South Africa" "biodiversity"',
                '"Sagole Baobab" "South Africa" "conservation"',
            ],
        )
        self.assertEqual(search_results_df["query_language"].unique().tolist(), ["en"])
        self.assertEqual(search_attempts_df["query_language"].unique().tolist(), ["en"])

    def test_builds_search_artifacts_for_dataframe(self):
        pilot_gdf = pd.DataFrame(
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
        )

        def fake_search(query: str, count: int, **_kwargs):
            return [
                {
                    "provider": "brave",
                    "title": "Search result",
                    "url": f"https://example.org/{query}",
                    "description": "Description",
                }
            ]

        search_results_df, search_attempts_df = build_location_topic_search_artifacts(
            pilot_gdf,
            search_func=fake_search,
            sleep_func=lambda _seconds: None,
            max_queries_per_polygon=4,
            results_per_query=1,
            request_delay_seconds=0,
        )

        self.assertEqual(len(search_results_df), 4)
        self.assertEqual(len(search_attempts_df), 4)


if __name__ == "__main__":
    unittest.main()

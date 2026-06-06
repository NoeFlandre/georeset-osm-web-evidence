import unittest
from collections import namedtuple

import pandas as pd

from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    add_pilot_metadata,
    attach_polygon_metadata,
    build_search_rows_for_query,
    build_candidate_urls,
    build_limited_localized_queries,
    query_languages_for_local_language,
    select_stratified_pilot_polygons,
    summarize_sentence_pilot,
)


class WorldwidePilotTests(unittest.TestCase):
    def test_selects_stratified_pilot_polygons_by_region_then_area_bin(self) -> None:
        rows = []
        for region in [
            "Africa",
            "Asia",
            "Europe",
            "North America",
            "Oceania",
            "South America",
        ]:
            for area_bin in ["tiny", "small", "medium", "large"]:
                for index in range(3):
                    rows.append(
                        {
                            "osm_type": "way",
                            "osm_id": f"{region}-{area_bin}-{index}",
                            "world_region": region,
                            "area_size_bin": area_bin,
                        }
                    )

        sample = select_stratified_pilot_polygons(
            pd.DataFrame(rows),
            sample_size=10,
            random_state=7,
        )

        self.assertEqual(len(sample), 10)
        self.assertEqual(set(sample["world_region"]), {
            "Africa",
            "Asia",
            "Europe",
            "North America",
            "Oceania",
            "South America",
        })
        self.assertGreaterEqual(sample["area_size_bin"].nunique(), 4)

    def test_adds_pilot_metadata_without_mutating_input(self) -> None:
        polygons_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "osm_tags": {"name": "Forêt Example", "landuse": "forest"},
                    "source_extract_id": "alsace",
                    "local_language": "en",
                }
            ]
        )

        result = add_pilot_metadata(polygons_df)

        self.assertNotIn("polygon_name", polygons_df.columns)
        self.assertEqual(result.loc[0, "polygon_name"], "Forêt Example")
        self.assertEqual(result.loc[0, "polygon_category"], "forest")
        self.assertEqual(result.loc[0, "query_local_language"], "fr")
        self.assertTrue(result["has_wikipedia_articles"].isna().all())

    def test_build_candidate_urls_keeps_wikipedia_and_caps_per_polygon(self) -> None:
        search_results_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "query": '"Forest A" forest',
                    "rank": 2,
                    "title": "A second result",
                    "url": "https://example.org/a",
                    "description": "Second",
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "query": '"Forest A" biodiversity',
                    "rank": 1,
                    "title": "Wikipedia result",
                    "url": "https://en.wikipedia.org/wiki/Forest_A",
                    "description": "Wikipedia",
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "query": '"Forest A" conservation',
                    "rank": 3,
                    "title": "A third result",
                    "url": "https://example.org/b",
                    "description": "Third",
                },
            ]
        )

        candidate_urls = build_candidate_urls(
            search_results_df,
            max_urls_per_polygon=2,
        )

        self.assertEqual(len(candidate_urls), 2)
        self.assertIn(
            "https://en.wikipedia.org/wiki/Forest_A",
            candidate_urls["url"].to_list(),
        )
        self.assertEqual(candidate_urls["best_rank"].to_list(), [1, 2])

    def test_build_search_rows_for_query_adds_query_and_polygon_metadata(self) -> None:
        Row = namedtuple(
            "Row",
            [
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "world_region",
                "country",
                "local_language",
                "query_local_language",
                "area_size_bin",
                "polygon_category",
            ],
        )
        polygon_row = Row(
            osm_type="way",
            osm_id=1,
            polygon_name="Forêt Example",
            has_wikipedia_articles=False,
            world_region="Europe",
            country="France",
            local_language="en",
            query_local_language="fr",
            area_size_bin="medium",
            polygon_category="forest",
        )
        results = [
            {
                "provider": "brave",
                "title": "Result title",
                "url": "https://example.org",
                "description": "Result description",
            }
        ]

        result_rows, attempt_row = build_search_rows_for_query(
            polygon_row=polygon_row,
            query_language="fr",
            query='"Forêt Example" forêt',
            results=results,
            search_error=None,
        )

        self.assertEqual(attempt_row["query_language"], "fr")
        self.assertIsNone(attempt_row["search_error"])
        self.assertEqual(attempt_row["result_count"], 1)
        self.assertEqual(result_rows[0]["rank"], 1)
        self.assertEqual(result_rows[0]["query_language"], "fr")
        self.assertEqual(result_rows[0]["world_region"], "Europe")
        self.assertEqual(result_rows[0]["query_local_language"], "fr")

    def test_attach_polygon_metadata_replaces_existing_metadata_without_suffixes(self) -> None:
        rows_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/page",
                    "world_region": "stale",
                    "query_local_language": "stale",
                }
            ]
        )
        pilot_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                }
            ]
        )

        result = attach_polygon_metadata(rows_df, pilot_df)

        self.assertEqual(result.loc[0, "url"], "https://example.org/page")
        self.assertEqual(result.loc[0, "world_region"], "Europe")
        self.assertEqual(result.loc[0, "query_local_language"], "fr")
        self.assertNotIn("world_region_x", result.columns)
        self.assertNotIn("world_region_y", result.columns)
        self.assertNotIn("query_local_language_x", result.columns)
        self.assertNotIn("query_local_language_y", result.columns)

    def test_summarizes_sentence_pilot_outputs(self) -> None:
        polygons_df = pd.DataFrame({"osm_id": [1, 2, 3]})
        page_text_df = pd.DataFrame(
            {
                "osm_id": [1, 1, 2],
                "url": ["a", "b", "c"],
                "fetch_error": [None, "403 Forbidden", None],
                "quality_score": [1.0, 0.0, 0.7],
            }
        )
        sentence_df = pd.DataFrame(
            {
                "osm_id": [1, 1, 2],
                "sentence": [
                    "One useful sentence.",
                    "Another useful sentence.",
                    "A third useful sentence.",
                ],
            }
        )

        summary = summarize_sentence_pilot(
            polygons_df=polygons_df,
            search_results_df=pd.DataFrame({"url": ["a", "b", "c", "d"]}),
            candidate_urls_df=pd.DataFrame({"url": ["a", "b", "c"]}),
            page_text_df=page_text_df,
            sentence_df=sentence_df,
        )

        self.assertEqual(summary["polygon_count"], 3)
        self.assertEqual(summary["search_result_count"], 4)
        self.assertEqual(summary["candidate_url_count"], 3)
        self.assertEqual(summary["successful_fetch_count"], 2)
        self.assertEqual(summary["high_quality_page_count"], 1)
        self.assertEqual(summary["sentence_count"], 3)
        self.assertEqual(summary["polygons_with_sentences"], 2)

    def test_query_languages_include_english_and_supported_local_language(self) -> None:
        self.assertEqual(
            query_languages_for_local_language(
                local_language="es",
                supported_languages={"en", "es", "fr"},
            ),
            ("en", "es"),
        )
        self.assertEqual(
            query_languages_for_local_language(
                local_language="en",
                supported_languages={"en", "es", "fr"},
            ),
            ("en",),
        )

    def test_limited_localized_queries_interleave_english_and_local_language(self) -> None:
        queries = build_limited_localized_queries(
            osm_tags={"name": "Rice", "landuse": "farmland"},
            local_language="si",
            supported_languages={"en", "si"},
            max_queries=4,
        )

        self.assertEqual([language for language, _ in queries], ["en", "si", "en", "si"])
        self.assertIn('"Rice" කෘෂිකර්මය', [query for _, query in queries])
        self.assertEqual(
            query_languages_for_local_language(
                local_language="xx",
                supported_languages={"en", "es", "fr"},
            ),
            ("en",),
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from scripts.evidence.build_english_only_sentence_pilot import (
    build_english_sentence_candidates,
    filter_english_candidate_urls,
)


class EnglishSentencePilotScriptTests(unittest.TestCase):
    def test_filter_english_candidate_urls_keeps_only_english_query_rows(self):
        candidate_urls_df = pd.DataFrame(
            [
                {"url": "https://example.org/en", "query_language": "en"},
                {"url": "https://example.org/fr", "query_language": "fr"},
            ]
        )

        result = filter_english_candidate_urls(candidate_urls_df)

        self.assertEqual(result["url"].to_list(), ["https://example.org/en"])

    def test_build_english_sentence_candidates_keeps_complete_english_polygons(self):
        page_rows = []
        for polygon_id in [1, 2]:
            for url_index in range(10):
                page_rows.append(
                    {
                        "osm_type": "way",
                        "osm_id": polygon_id,
                        "polygon_name": f"Forest {polygon_id}",
                        "has_wikipedia_articles": False,
                        "url": f"https://example.org/{polygon_id}/{url_index}",
                        "final_url": f"https://example.org/{polygon_id}/{url_index}",
                        "search_title": "Search title",
                        "search_description": "Search description",
                        "search_queries": '"Forest" biodiversity',
                        "title": "Page title",
                        "text_length": 200,
                        "quality_score": 1.0,
                        "quality_flags": [],
                        "query_language": "en",
                        "text": (
                            "The forest contains wetlands and protected wildlife "
                            f"habitats number {url_index}."
                        ),
                        "query_local_language": "en",
                    }
                )
        page_rows.append(
            {
                "osm_type": "way",
                "osm_id": 3,
                "polygon_name": "Forêt",
                "has_wikipedia_articles": False,
                "url": "https://example.org/fr",
                "final_url": "https://example.org/fr",
                "search_title": "Search title",
                "search_description": "Search description",
                "search_queries": '"Forêt" biodiversité',
                "title": "Page title",
                "text_length": 200,
                "quality_score": 1.0,
                "quality_flags": [],
                "query_language": "fr",
                "text": "Cette forêt contient des zones humides protégées.",
                "query_local_language": "fr",
            }
        )
        page_text_with_quality_df = pd.DataFrame(page_rows)
        pilot_df = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "world_region": "Europe"},
                {"osm_type": "way", "osm_id": 2, "world_region": "North America"},
                {"osm_type": "way", "osm_id": 3, "world_region": "Europe"},
            ]
        )

        result = build_english_sentence_candidates(
            page_text_with_quality_df,
            pilot_df,
            target_polygon_count=2,
            sentences_per_polygon=10,
            sentences_per_url=1,
        )

        self.assertEqual(len(result), 20)
        self.assertEqual(set(result["query_language"]), {"en"})
        self.assertEqual(result.groupby(["osm_type", "osm_id"]).size().to_list(), [10, 10])
        self.assertTrue(result.groupby(["osm_type", "osm_id", "url"]).size().eq(1).all())


if __name__ == "__main__":
    unittest.main()

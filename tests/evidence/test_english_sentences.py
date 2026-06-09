import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.english_sentences import (
    build_english_sentence_candidates,
)


class EnglishSentenceCandidateTests(unittest.TestCase):
    def test_build_english_sentence_candidates_keeps_complete_english_polygons(self):
        page_rows = []
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
        polygon_terms = {1: "northern", 2: "southern"}
        for polygon_id in [1, 2]:
            for url_index, sentence_term in enumerate(sentence_terms):
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
                            f"The {polygon_terms[polygon_id]} {sentence_term} "
                            "forest contains wetlands and protected wildlife habitats."
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

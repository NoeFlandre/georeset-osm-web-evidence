import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.sentence_candidates import (
    build_sentence_candidate_dataframe,
)


class TestEvidenceSentenceCandidate(unittest.TestCase):
    def test_builds_one_row_per_sentence_candidate(self):
        text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 123,
                    "polygon_name": "Forêt Test",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/page",
                    "final_url": "https://example.com/final",
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "search_queries": '"Forêt Test" biodiversité',
                    "title": "Page title",
                    "text_length": 200,
                    "quality_score": 1.0,
                    "quality_flags": [],
                    "text": (
                        "Home. "
                        "This sentence contains enough words to become a useful candidate. "
                        "Map. "
                        "Another sentence contains enough useful words for labeling."
                    ),
                },
                {
                    "osm_type": "way",
                    "osm_id": 456,
                    "polygon_name": "Empty Text",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/empty",
                    "final_url": "https://example.com/empty",
                    "search_title": "Empty",
                    "search_description": "Empty",
                    "search_queries": '"Empty" biodiversité',
                    "title": "Empty title",
                    "text_length": 0,
                    "quality_score": 0.0,
                    "quality_flags": ["empty_text"],
                    "text": None,
                },
            ]
        )

        sentence_df = build_sentence_candidate_dataframe(text_df)

        self.assertEqual(len(sentence_df), 2)
        self.assertEqual(
            sentence_df["polygon_name"].to_list(), ["Forêt Test", "Forêt Test"]
        )
        self.assertEqual(sentence_df["osm_id"].to_list(), [123, 123])
        self.assertEqual(
            sentence_df["sentence"].to_list(),
            [
                "This sentence contains enough words to become a useful candidate.",
                "Another sentence contains enough useful words for labeling.",
            ],
        )
        self.assertIn("quality_score", sentence_df.columns)
        self.assertIn("search_queries", sentence_df.columns)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.final_url_artifacts import (
    select_exact_url_artifacts,
    validate_exact_sentence_url_counts,
)


class FinalUrlArtifactTests(unittest.TestCase):
    def test_selects_exact_url_artifacts_from_sentence_urls(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": f"https://example.org/{index}",
                    "sentence": f"Sentence {index} contains enough words for testing.",
                }
                for index in range(10)
            ]
        )
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": f"https://example.org/{index}",
                    "best_rank": index,
                }
                for index in range(12)
            ]
        )
        page_text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "source_url": f"https://example.org/{index}",
                    "text": "Page text",
                }
                for index in range(12)
            ]
        )

        selected_urls_df, selected_page_text_df = select_exact_url_artifacts(
            sentence_df=sentence_df,
            candidate_urls_df=candidate_urls_df,
            page_text_df=page_text_df,
            urls_per_polygon=10,
        )

        self.assertEqual(len(selected_urls_df), 10)
        self.assertEqual(len(selected_page_text_df), 10)
        self.assertEqual(selected_urls_df["url"].to_list(), sentence_df["url"].to_list())

    def test_validate_exact_sentence_url_counts_rejects_duplicate_url_selection(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "First valid sentence about the forest habitat.",
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "Second valid sentence about the forest habitat.",
                },
            ]
        )

        with self.assertRaises(ValueError):
            validate_exact_sentence_url_counts(sentence_df, urls_per_polygon=2)


if __name__ == "__main__":
    unittest.main()

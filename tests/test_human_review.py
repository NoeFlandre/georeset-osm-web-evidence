import unittest

import pandas as pd

from georeset_osm_web_evidence.review.human import (
    build_human_review_dataframe,
    select_successful_review_rows,
)


class HumanReviewTests(unittest.TestCase):
    def test_builds_reviewer_friendly_rows_from_successful_fetches(self):
        source = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 123,
                    "polygon_name": "Forêt test",
                    "has_wikipedia_articles": False,
                    "source_url": "https://example.test/forest",
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "title": "Page title",
                    "text": "word " * 500,
                    "text_length": 2500,
                    "fetch_error": None,
                },
                {
                    "osm_type": "way",
                    "osm_id": 456,
                    "polygon_name": "Marais test",
                    "has_wikipedia_articles": True,
                    "source_url": "https://example.test/broken",
                    "search_title": "Broken result",
                    "search_description": "Broken description",
                    "title": None,
                    "text": None,
                    "text_length": 0,
                    "fetch_error": "403 Forbidden",
                },
            ]
        )

        review_df = build_human_review_dataframe(source, preview_chars=40)

        self.assertEqual(
            list(review_df.columns[:4]),
            ["review_id", "human_label", "human_notes", "fetch_status"],
        )
        self.assertEqual(review_df.loc[0, "review_id"], "review-0001")
        self.assertEqual(review_df.loc[0, "fetch_status"], "fetched")
        self.assertEqual(review_df.loc[0, "human_label"], "")
        self.assertEqual(review_df.loc[0, "human_notes"], "")
        self.assertLessEqual(len(review_df.loc[0, "text_preview"]), 41)
        self.assertEqual(len(review_df), 1)

    def test_selects_successful_rows_with_caps_per_polygon(self):
        source = pd.DataFrame(
            [
                {
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": False,
                    "osm_type": "way",
                    "source_url": "https://example.test/a1",
                    "text": "usable",
                    "text_length": 100,
                    "fetch_error": None,
                },
                {
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": False,
                    "osm_type": "way",
                    "source_url": "https://example.test/a2",
                    "text": "usable",
                    "text_length": 100,
                    "fetch_error": None,
                },
                {
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": False,
                    "osm_type": "way",
                    "source_url": "https://example.test/a3",
                    "text": "usable",
                    "text_length": 100,
                    "fetch_error": None,
                },
                {
                    "polygon_name": "Marsh B",
                    "has_wikipedia_articles": True,
                    "osm_type": "relation",
                    "source_url": "https://example.test/b1",
                    "text": "usable",
                    "text_length": 100,
                    "fetch_error": None,
                },
                {
                    "polygon_name": "Broken C",
                    "has_wikipedia_articles": True,
                    "osm_type": "way",
                    "source_url": "https://example.test/c1",
                    "text": None,
                    "text_length": 0,
                    "fetch_error": "403 Forbidden",
                },
            ]
        )

        selected = select_successful_review_rows(
            source,
            max_rows=10,
            max_rows_per_polygon=2,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected["polygon_name"].value_counts()["Forest A"], 2)
        self.assertNotIn("Broken C", selected["polygon_name"].to_list())


if __name__ == "__main__":
    unittest.main()

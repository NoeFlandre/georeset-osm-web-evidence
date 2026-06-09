import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from openpyxl import load_workbook

from georeset_osm_web_evidence.review.sentence_labels import (
    build_sentence_label_review_dataframe,
    save_sentence_label_review_xlsx,
)


class SentenceLabelReviewTests(unittest.TestCase):
    def test_builds_review_rows_with_llm_label_and_empty_human_columns(self):
        labeled_df = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "llm_label": "relevant",
                    "parse_error": None,
                    "sentence": "This reserve contains visible wetland habitat.",
                    "polygon_name": "Sagole Baobab",
                    "url": "https://example.org/a",
                    "title": "Page title",
                    "search_queries": '"Sagole Baobab" "South Africa" protected area',
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "country": "South Africa",
                    "world_region": "Africa",
                    "polygon_category": "protected_area",
                    "area_size_bin": "tiny",
                    "quality_score": 1.0,
                    "osm_type": "way",
                    "osm_id": 1,
                }
            ]
        )

        review_df = build_sentence_label_review_dataframe(labeled_df)

        self.assertEqual(review_df.loc[0, "review_id"], "sentence-review-0001")
        self.assertEqual(review_df.loc[0, "human_label"], "")
        self.assertEqual(review_df.loc[0, "human_notes"], "")
        self.assertEqual(review_df.loc[0, "llm_label"], "relevant")
        self.assertEqual(
            review_df.loc[0, "sentence"],
            "This reserve contains visible wetland habitat.",
        )
        self.assertEqual(review_df.loc[0, "source_url"], "https://example.org/a")

    def test_builds_review_rows_from_existing_review_url_columns(self):
        labeled_df = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "llm_label": "irrelevant",
                    "parse_error": None,
                    "sentence": "This sentence is already in the review schema.",
                    "polygon_name": "Bird Island Nature Reserve",
                    "source_url": "https://example.org/b",
                    "page_title": "Existing title",
                    "search_queries": '"Bird Island Nature Reserve" "South Africa" wetland',
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "country": "South Africa",
                    "world_region": "Africa",
                    "polygon_category": "wetland",
                    "area_size_bin": "tiny",
                    "quality_score": 0.9,
                    "osm_type": "way",
                    "osm_id": 3,
                }
            ]
        )

        review_df = build_sentence_label_review_dataframe(labeled_df)

        self.assertEqual(review_df.loc[0, "source_url"], "https://example.org/b")
        self.assertEqual(review_df.loc[0, "page_title"], "Existing title")

    def test_saves_sentence_label_review_workbook(self):
        review_df = pd.DataFrame(
            [
                {
                    "review_id": "sentence-review-0001",
                    "human_label": "",
                    "human_notes": "",
                    "llm_label": "irrelevant",
                    "sentence": "A sentence for review.",
                    "polygon_name": "Park Wood",
                    "source_url": "https://example.org/a",
                    "page_title": "Page title",
                    "search_queries": '"Park Wood" "Essex" forest',
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "country": "Essex",
                    "world_region": "Europe",
                    "polygon_category": "forest",
                    "area_size_bin": "small",
                    "quality_score": 0.8,
                    "osm_type": "way",
                    "osm_id": 2,
                    "sentence_id": "s1",
                    "parse_error": "",
                }
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sentence_review.xlsx"
            save_sentence_label_review_xlsx(review_df, path)
            workbook = load_workbook(path)
            worksheet = workbook["review"]

        self.assertEqual(worksheet.freeze_panes, "E2")
        self.assertEqual(worksheet["B1"].value, "human_label")
        self.assertEqual(worksheet["D1"].value, "llm_label")
        self.assertEqual(worksheet["E1"].value, "sentence")
        self.assertEqual(worksheet["G1"].value, "source_url")
        self.assertEqual(worksheet["G2"].hyperlink.target, "https://example.org/a")


if __name__ == "__main__":
    unittest.main()

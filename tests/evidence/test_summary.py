import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.summary import summarize_polygon_evidence


class EvidenceSummaryTests(unittest.TestCase):
    def test_summarizes_page_evidence_per_polygon(self):
        polygons_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": False,
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Forest B",
                    "has_wikipedia_articles": True,
                },
            ]
        )

        page_text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "source_url": "https://example.test/a",
                    "fetch_error": None,
                    "quality_score": 1.0,
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "source_url": "https://example.test/b",
                    "fetch_error": "403 Forbidden",
                    "quality_score": 0.0,
                },
            ]
        )

        summary = summarize_polygon_evidence(
            polygons_df,
            page_text_df,
        )

        self.assertEqual(len(summary), 2)

        forest_a = summary[summary["osm_id"] == 1].iloc[0]
        forest_b = summary[summary["osm_id"] == 2].iloc[0]

        self.assertEqual(forest_a["candidate_url_count"], 2)
        self.assertEqual(forest_a["successful_fetch_count"], 1)
        self.assertEqual(forest_a["high_quality_page_count"], 1)
        self.assertEqual(forest_a["max_quality_score"], 1.0)
        self.assertTrue(forest_a["has_high_quality_evidence"])

        self.assertEqual(forest_b["candidate_url_count"], 0)
        self.assertEqual(forest_b["successful_fetch_count"], 0)
        self.assertEqual(forest_b["max_quality_score"], 0.0)
        self.assertEqual(forest_b["high_quality_page_count"], 0)
        self.assertFalse(forest_b["has_high_quality_evidence"])

    def test_counts_unique_candidate_urls_and_inclusive_quality_threshold(self):
        polygons_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": False,
                },
            ]
        )
        page_text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "source_url": "https://example.test/a",
                    "fetch_error": None,
                    "quality_score": 0.8,
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "source_url": "https://example.test/a",
                    "fetch_error": None,
                    "quality_score": 0.7,
                },
            ]
        )

        summary = summarize_polygon_evidence(
            polygons_df,
            page_text_df,
            high_quality_threshold=0.8,
        )

        row = summary.iloc[0]
        self.assertEqual(row["candidate_url_count"], 1)
        self.assertEqual(row["successful_fetch_count"], 2)
        self.assertEqual(row["high_quality_page_count"], 1)
        self.assertTrue(row["has_high_quality_evidence"])


if __name__ == "__main__":
    unittest.main()

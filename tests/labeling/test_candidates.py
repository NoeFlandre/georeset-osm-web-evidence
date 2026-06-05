import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from georeset_osm_web_evidence.evidence.labeling_candidates import (
    build_labeling_candidates,
    write_labeling_candidates_jsonl,
)


class TestLabelingCandidates(unittest.TestCase):
    def test_builds_clean_deduplicated_labeling_candidates(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forêt Alpha",
                    "url": "https://example.com/a",
                    "quality_score": 1.0,
                    "sentence": "  This forest contains wetlands and mixed oak stands. ",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Forêt Beta",
                    "url": "https://example.com/b",
                    "quality_score": 0.9,
                    "sentence": "This forest contains wetlands and mixed oak stands.",
                },
                {
                    "osm_type": "way",
                    "osm_id": 3,
                    "polygon_name": "Marais Gamma",
                    "url": "https://example.com/c",
                    "quality_score": 0.5,
                    "sentence": "This low quality sentence should be filtered out.",
                },
                {
                    "osm_type": "relation",
                    "osm_id": 4,
                    "polygon_name": "Réserve Delta",
                    "url": "https://example.com/d",
                    "quality_score": 1.0,
                    "sentence": None,
                },
                {
                    "osm_type": "relation",
                    "osm_id": 5,
                    "polygon_name": "Plaine Epsilon",
                    "url": "https://example.com/e",
                    "quality_score": 0.8,
                    "sentence": "This plain includes irrigated crops and open grassland.",
                },
            ]
        )

        labeling_df = build_labeling_candidates(sentence_df, min_quality_score=0.8)
        labeling_df_again = build_labeling_candidates(sentence_df, min_quality_score=0.8)

        self.assertEqual(len(labeling_df), 2)
        self.assertEqual(
            labeling_df["model_input"].to_list(),
            [
                "This forest contains wetlands and mixed oak stands.",
                "This plain includes irrigated crops and open grassland.",
            ],
        )
        self.assertEqual(
            labeling_df["sentence_id"].to_list(),
            labeling_df_again["sentence_id"].to_list(),
        )
        self.assertTrue(labeling_df["sentence_id"].str.len().eq(64).all())
        self.assertTrue((labeling_df["quality_score"] >= 0.8).all())
        self.assertIn("polygon_name", labeling_df.columns)
        self.assertIn("url", labeling_df.columns)

    def test_writes_minimal_jsonl_for_llm_labeling(self):
        labeling_df = pd.DataFrame(
            [
                {"sentence_id": "abc", "model_input": "First useful sentence."},
                {"sentence_id": "def", "model_input": "Second useful sentence."},
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "labeling_candidates.jsonl"
            write_labeling_candidates_jsonl(labeling_df, output_path)

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            rows,
            [
                {"sentence_id": "abc", "text": "First useful sentence."},
                {"sentence_id": "def", "text": "Second useful sentence."},
            ],
        )


if __name__ == "__main__":
    unittest.main()

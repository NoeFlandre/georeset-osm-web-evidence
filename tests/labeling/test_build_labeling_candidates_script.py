import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import scripts.evidence.build_labeling_candidates as labeling_candidate_script
from scripts.evidence.build_labeling_candidates import run_labeling_candidate_build


class TestBuildLabelingCandidatesScript(unittest.TestCase):
    def test_builds_parquet_and_jsonl_outputs(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "polygon_name": "Forêt Alpha",
                    "url": "https://example.com/a",
                    "quality_score": 1.0,
                    "sentence": "This forest contains wetlands and mixed oak stands.",
                },
                {
                    "polygon_name": "Marais Beta",
                    "url": "https://example.com/b",
                    "quality_score": 0.5,
                    "sentence": "This sentence should be filtered out.",
                },
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "sentence_candidates.parquet"
            parquet_output_path = Path(temporary_directory) / "labeling.parquet"
            jsonl_output_path = Path(temporary_directory) / "labeling.jsonl"
            sentence_df.to_parquet(input_path, index=False)

            labeling_df = run_labeling_candidate_build(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
                min_quality_score=0.8,
            )

            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_rows = [
                json.loads(line)
                for line in jsonl_output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(labeling_df), 1)
        self.assertEqual(len(saved_df), 1)
        self.assertEqual(jsonl_rows[0]["sentence_id"], saved_df.loc[0, "sentence_id"])
        self.assertEqual(jsonl_rows[0]["text"], saved_df.loc[0, "model_input"])

    def test_formats_labeling_candidate_summary(self):
        labeling_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.com/a",
                    "quality_score": 1.0,
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.com/b",
                    "quality_score": 0.8,
                },
                {
                    "osm_type": "relation",
                    "osm_id": 2,
                    "url": "https://example.com/b",
                    "quality_score": 0.6,
                },
            ]
        )

        self.assertTrue(
            hasattr(labeling_candidate_script, "format_labeling_candidate_summary")
        )
        summary = labeling_candidate_script.format_labeling_candidate_summary(
            labeling_df=labeling_df,
            parquet_output_path=Path("out") / "candidates.parquet",
            jsonl_output_path=Path("out") / "candidates.jsonl",
        )

        self.assertEqual(
            summary,
            "Saved 3 labeling candidates to out/candidates.parquet\n"
            "Saved JSONL inputs to out/candidates.jsonl\n"
            "Covered 2 polygons\n"
            "Covered 2 URLs\n"
            "Mean quality score: 0.800",
        )


if __name__ == "__main__":
    unittest.main()

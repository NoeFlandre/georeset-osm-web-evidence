import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()

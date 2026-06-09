import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.evidence import build_english_context_query_sentence_pilot as pilot


class EnglishContextQueryPilotScriptTests(unittest.TestCase):
    def test_experiment_paths_do_not_overwrite_english_only_pilot(self):
        self.assertEqual(
            pilot.OUTPUT_DIR,
            Path(
                "data/processed/pilots/"
                "worldwide_sentence_pilot_10_english_context_queries_v1"
            ),
        )
        self.assertNotEqual(pilot.OUTPUT_DIR, pilot.ENGLISH_ONLY_OUTPUT_DIR)

    def test_writes_context_pilot_prompt_request_artifacts(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "This nature reserve contains visible wetland habitat.",
                    "polygon_name": "Sagole Baobab",
                    "query_language": "en",
                }
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "sentence_candidates.parquet"
            parquet_output_path = temp_path / "llm_requests.parquet"
            jsonl_output_path = temp_path / "llm_requests.jsonl"
            sentence_df.to_parquet(input_path, index=False)

            prompt_df = pilot.run_context_pilot_labeling_request_build(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
            )

            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_exists = jsonl_output_path.exists()

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(len(saved_df), 1)
        self.assertTrue(jsonl_exists)
        self.assertEqual(saved_df.loc[0, "model_input"], sentence_df.loc[0, "sentence"])


if __name__ == "__main__":
    unittest.main()

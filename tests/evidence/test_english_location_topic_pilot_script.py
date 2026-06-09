import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.evidence import build_english_location_topic_sentence_pilot as pilot


class EnglishLocationTopicPilotScriptTests(unittest.TestCase):
    def test_experiment_paths_are_isolated(self):
        self.assertEqual(
            pilot.OUTPUT_DIR,
            Path(
                "data/processed/pilots/"
                "worldwide_sentence_pilot_10_english_location_topic_queries_v1"
            ),
        )
        self.assertNotEqual(pilot.OUTPUT_DIR, pilot.ENGLISH_ONLY_OUTPUT_DIR)

    def test_writes_location_aware_prompt_artifacts(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "The reserve contains extensive wetland habitat.",
                    "polygon_name": "Sagole Baobab",
                    "country": "South Africa",
                    "world_region": "Africa",
                    "polygon_category": "protected_area",
                    "title": "Sagole Baobab travel guide",
                    "search_queries": '"Sagole Baobab" "South Africa" "protected area"',
                }
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "sentence_candidates.parquet"
            parquet_output_path = temp_path / "llm_requests.parquet"
            jsonl_output_path = temp_path / "llm_requests.jsonl"
            sentence_df.to_parquet(input_path, index=False)

            prompt_df = pilot.run_location_topic_labeling_request_build(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
            )

            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_exists = jsonl_output_path.exists()

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(len(saved_df), 1)
        self.assertTrue(jsonl_exists)
        self.assertIn("specific target polygon", saved_df.loc[0, "prompt"])


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from georeset_osm_web_evidence.labeling.parser import parse_binary_label_response
from georeset_osm_web_evidence.labeling.prompt import (
    PROMPT_VERSION,
    build_binary_label_prompt,
)
from georeset_osm_web_evidence.labeling.requests import (
    build_labeling_prompt_rows,
    write_labeling_prompt_jsonl,
)
from scripts.labeling.build_labeling_prompt_sample import (
    run_labeling_prompt_sample_build,
)


class LabelingPromptScaffoldTests(unittest.TestCase):
    def test_builds_binary_prompt_without_extra_output_fields(self):
        prompt = build_binary_label_prompt(
            "The forest contains wetlands visible from satellite imagery."
        )

        self.assertIn("remote sensing", prompt)
        self.assertIn("relevant", prompt)
        self.assertIn("irrelevant", prompt)
        self.assertIn("The forest contains wetlands", prompt)
        self.assertIn("Reply with exactly one word", prompt)
        self.assertNotIn("rationale", prompt.lower())
        self.assertNotIn("confidence", prompt.lower())

    def test_parses_only_binary_labels(self):
        self.assertEqual(parse_binary_label_response(" Relevant \n"), "relevant")
        self.assertEqual(parse_binary_label_response('"irrelevant".'), "irrelevant")

        with self.assertRaises(ValueError):
            parse_binary_label_response("relevant because wetlands are visible")

        with self.assertRaises(ValueError):
            parse_binary_label_response("maybe")

    def test_builds_prompt_rows_from_labeling_candidates(self):
        labeling_df = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "model_input": "This wetland contains reed beds.",
                    "polygon_name": "Marais Alpha",
                    "quality_score": 0.9,
                },
                {
                    "sentence_id": "s2",
                    "model_input": "Opening hours are available online.",
                    "polygon_name": "Bois Beta",
                    "quality_score": 0.8,
                },
            ]
        )

        prompt_df = build_labeling_prompt_rows(labeling_df, limit=1)

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(prompt_df.loc[0, "sentence_id"], "s1")
        self.assertEqual(prompt_df.loc[0, "prompt_version"], PROMPT_VERSION)
        self.assertIn("This wetland contains reed beds.", prompt_df.loc[0, "prompt"])
        self.assertEqual(prompt_df.loc[0, "llm_label"], None)
        self.assertEqual(prompt_df.loc[0, "raw_response"], None)
        self.assertEqual(prompt_df.loc[0, "parse_error"], None)
        self.assertEqual(prompt_df.loc[0, "polygon_name"], "Marais Alpha")

    def test_writes_prompt_jsonl_and_script_outputs(self):
        labeling_df = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "model_input": "This wetland contains reed beds.",
                    "polygon_name": "Marais Alpha",
                },
                {
                    "sentence_id": "s2",
                    "model_input": "This forest has dense canopy cover.",
                    "polygon_name": "Bois Beta",
                },
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "labeling_candidates.parquet"
            parquet_output_path = temp_path / "prompt_sample.parquet"
            jsonl_output_path = temp_path / "prompt_sample.jsonl"
            labeling_df.to_parquet(input_path, index=False)

            prompt_df = run_labeling_prompt_sample_build(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
                sample_size=2,
            )
            write_labeling_prompt_jsonl(prompt_df.head(1), temp_path / "one.jsonl")

            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_rows = [
                json.loads(line)
                for line in jsonl_output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(saved_df), 2)
        self.assertEqual(len(jsonl_rows), 2)
        self.assertEqual(
            sorted(jsonl_rows[0]),
            ["prompt", "prompt_version", "sentence_id"],
        )
        self.assertEqual(jsonl_rows[0]["sentence_id"], saved_df.loc[0, "sentence_id"])


if __name__ == "__main__":
    unittest.main()

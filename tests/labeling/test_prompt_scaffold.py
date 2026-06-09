import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from georeset_osm_web_evidence.labeling.parser import parse_binary_label_response
from georeset_osm_web_evidence.labeling.prompt import (
    LOCATION_AWARE_PROMPT_VERSION,
    PROMPT_VERSION,
    build_location_aware_binary_label_prompt,
    build_binary_label_prompt,
)
from georeset_osm_web_evidence.labeling.requests import (
    build_and_write_labeling_prompt_artifacts,
    build_location_aware_sentence_candidate_prompt_rows,
    build_sentence_candidate_prompt_rows,
    build_labeling_prompt_rows,
    write_labeling_prompt_jsonl,
)
from scripts.labeling.build_labeling_prompt_sample import (
    run_labeling_prompt_sample_build,
)
from scripts.labeling.build_english_pilot_labeling_requests import (
    run_english_pilot_labeling_request_build,
)


class LabelingPromptScaffoldTests(unittest.TestCase):
    def test_builds_binary_prompt_without_extra_output_fields(self):
        prompt = build_binary_label_prompt(
            "The forest contains wetlands visible from satellite imagery."
        )

        self.assertIn("valid JSON", prompt)
        self.assertIn("remote sensing", prompt)
        self.assertIn("relevant", prompt)
        self.assertIn("irrelevant", prompt)
        self.assertIn("The forest contains wetlands", prompt)
        self.assertIn('{"label":"relevant"}', prompt)
        self.assertNotIn("rationale", prompt.lower())
        self.assertNotIn("confidence", prompt.lower())

    def test_builds_location_aware_prompt_for_specific_polygon_relevance(self):
        prompt = build_location_aware_binary_label_prompt(
            sentence="The reserve contains extensive wetland habitat.",
            polygon_name="Sagole Baobab",
            location_context="South Africa, Africa",
            polygon_category="protected_area",
            page_title="Sagole Baobab travel guide",
            search_query='"Sagole Baobab" "South Africa" "protected area"',
        )

        self.assertIn("Target polygon", prompt)
        self.assertIn("Sagole Baobab", prompt)
        self.assertIn("South Africa, Africa", prompt)
        self.assertIn("specific target polygon", prompt)
        self.assertIn("generic fact", prompt)
        self.assertIn("remote-sensing", prompt)
        self.assertNotIn("Use only the sentence", prompt)
        self.assertIn('{"label":"relevant"}', prompt)
        self.assertIn('{"label":"irrelevant"}', prompt)
        self.assertNotIn("rationale", prompt.lower())
        self.assertNotIn("confidence", prompt.lower())

    def test_rejects_empty_or_non_string_prompt_sentence(self):
        with self.assertRaises(ValueError):
            build_binary_label_prompt("")

        with self.assertRaises(ValueError):
            build_binary_label_prompt(None)

    def test_parses_only_binary_labels(self):
        self.assertEqual(
            parse_binary_label_response('{"label": "relevant"}'),
            "relevant",
        )
        self.assertEqual(
            parse_binary_label_response('```json\n{"label":"irrelevant"}\n```'),
            "irrelevant",
        )

        with self.assertRaises(ValueError):
            parse_binary_label_response("relevant")

        with self.assertRaises(ValueError):
            parse_binary_label_response('{"label": "relevant", "rationale": "wetlands"}')

        with self.assertRaises(ValueError):
            parse_binary_label_response('{"label": "unclear"}')

        with self.assertRaises(ValueError):
            parse_binary_label_response("relevant because wetlands are visible")

        with self.assertRaises(ValueError):
            parse_binary_label_response("maybe")

        with self.assertRaises(ValueError):
            parse_binary_label_response("relevant!")

        with self.assertRaises(ValueError):
            parse_binary_label_response(None)

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

    def test_builds_prompt_rows_directly_from_sentence_candidates_without_deduping(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "This wetland contains reed beds.",
                    "polygon_name": "Marais Alpha",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "url": "https://example.org/b",
                    "sentence": "This wetland contains reed beds.",
                    "polygon_name": "Marais Beta",
                },
            ]
        )

        prompt_df = build_sentence_candidate_prompt_rows(sentence_df)

        self.assertEqual(len(prompt_df), 2)
        self.assertEqual(
            prompt_df["model_input"].to_list(),
            [
                "This wetland contains reed beds.",
                "This wetland contains reed beds.",
            ],
        )
        self.assertNotEqual(
            prompt_df.loc[0, "sentence_id"],
            prompt_df.loc[1, "sentence_id"],
        )
        self.assertIn("This wetland contains reed beds.", prompt_df.loc[0, "prompt"])
        self.assertEqual(prompt_df["llm_label"].to_list(), [None, None])

    def test_builds_location_aware_prompt_rows_from_sentence_candidates(self):
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

        prompt_df = build_location_aware_sentence_candidate_prompt_rows(sentence_df)

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(
            prompt_df.loc[0, "prompt_version"],
            LOCATION_AWARE_PROMPT_VERSION,
        )
        self.assertIn("Sagole Baobab", prompt_df.loc[0, "prompt"])
        self.assertIn("South Africa, Africa", prompt_df.loc[0, "prompt"])
        self.assertEqual(prompt_df.loc[0, "llm_label"], None)

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

    def test_builds_english_pilot_prompt_request_artifacts(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "This forest contains dense evergreen canopy.",
                    "polygon_name": "Forest Alpha",
                    "query_language": "en",
                }
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "english_sentence_candidates.parquet"
            parquet_output_path = temp_path / "llm_requests.parquet"
            jsonl_output_path = temp_path / "llm_requests.jsonl"
            sentence_df.to_parquet(input_path, index=False)

            prompt_df = run_english_pilot_labeling_request_build(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
            )
            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_rows = [
                json.loads(line)
                for line in jsonl_output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(len(saved_df), 1)
        self.assertEqual(len(jsonl_rows), 1)
        self.assertEqual(saved_df.loc[0, "model_input"], sentence_df.loc[0, "sentence"])
        self.assertEqual(saved_df.loc[0, "llm_label"], None)

    def test_builds_and_writes_prompt_artifacts_with_supplied_builder(self):
        source_df = pd.DataFrame(
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
            input_path = temp_path / "input" / "labeling_candidates.parquet"
            parquet_output_path = temp_path / "requests" / "llm_requests.parquet"
            jsonl_output_path = temp_path / "requests" / "llm_requests.jsonl"
            input_path.parent.mkdir(parents=True)
            source_df.to_parquet(input_path, index=False)

            prompt_df = build_and_write_labeling_prompt_artifacts(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
                prompt_builder=lambda dataframe: build_labeling_prompt_rows(
                    dataframe,
                    limit=1,
                ),
            )

            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_rows = [
                json.loads(line)
                for line in jsonl_output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(len(saved_df), 1)
        self.assertEqual(len(jsonl_rows), 1)
        self.assertEqual(prompt_df.loc[0, "sentence_id"], "s1")
        self.assertEqual(saved_df.loc[0, "sentence_id"], "s1")
        self.assertEqual(jsonl_rows[0]["sentence_id"], "s1")


if __name__ == "__main__":
    unittest.main()

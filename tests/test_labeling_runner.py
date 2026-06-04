import unittest

import pandas as pd

from georeset_osm_web_evidence.labeling.runner import label_prompt_rows


class LabelingRunnerTests(unittest.TestCase):
    def test_labels_prompt_rows_with_provider_function(self):
        prompt_df = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "prompt": "Prompt for wetlands",
                    "polygon_name": "Marais Alpha",
                },
                {
                    "sentence_id": "s2",
                    "prompt": "Prompt for opening hours",
                    "polygon_name": "Bois Beta",
                },
            ]
        )
        responses = {
            "Prompt for wetlands": "relevant",
            "Prompt for opening hours": "irrelevant",
        }
        seen_prompts = []

        def fake_label_fn(prompt: str) -> str:
            seen_prompts.append(prompt)
            return responses[prompt]

        labeled_df = label_prompt_rows(prompt_df, fake_label_fn)

        self.assertEqual(seen_prompts, prompt_df["prompt"].to_list())
        self.assertEqual(labeled_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(labeled_df["raw_response"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(labeled_df["parse_error"].to_list(), [None, None])
        self.assertEqual(labeled_df["polygon_name"].to_list(), ["Marais Alpha", "Bois Beta"])

    def test_records_parse_and_provider_errors_without_stopping(self):
        prompt_df = pd.DataFrame(
            [
                {"sentence_id": "s1", "prompt": "good"},
                {"sentence_id": "s2", "prompt": "messy"},
                {"sentence_id": "s3", "prompt": "crash"},
            ]
        )

        def fake_label_fn(prompt: str) -> str:
            if prompt == "good":
                return "relevant"
            if prompt == "messy":
                return "maybe relevant"
            raise RuntimeError("provider unavailable")

        labeled_df = label_prompt_rows(prompt_df, fake_label_fn)

        self.assertEqual(labeled_df.loc[0, "llm_label"], "relevant")
        self.assertEqual(labeled_df.loc[0, "parse_error"], None)

        self.assertEqual(labeled_df.loc[1, "raw_response"], "maybe relevant")
        self.assertEqual(labeled_df.loc[1, "llm_label"], None)
        self.assertIn("Expected exactly one label", labeled_df.loc[1, "parse_error"])

        self.assertEqual(labeled_df.loc[2, "raw_response"], None)
        self.assertEqual(labeled_df.loc[2, "llm_label"], None)
        self.assertIn("RuntimeError", labeled_df.loc[2, "parse_error"])


if __name__ == "__main__":
    unittest.main()

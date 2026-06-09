import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import scripts.evidence.extract_sentence_candidate as extract_script


class ExtractSentenceCandidateScriptTests(unittest.TestCase):
    def test_builds_sentence_candidate_artifact(self):
        page_text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Bois Alpha",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/a",
                    "final_url": "https://example.com/a",
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "search_queries": "query A",
                    "title": "Fetched page title",
                    "text_length": 120,
                    "quality_score": 0.9,
                    "quality_flags": [],
                    "query_language": "en",
                    "text": (
                        "This forest contains mixed canopy and wetland habitat "
                        "visible from satellite imagery."
                    ),
                }
            ],
            index=[17],
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "page_text.parquet"
            output_path = temp_path / "nested" / "sentences.parquet"
            page_text_df.to_parquet(input_path, index=False)

            self.assertTrue(hasattr(extract_script, "run_sentence_candidate_build"))
            sentence_df = extract_script.run_sentence_candidate_build(
                input_path=input_path,
                output_path=output_path,
            )

            saved_df = pd.read_parquet(output_path)

        self.assertEqual(len(sentence_df), 1)
        self.assertEqual(saved_df.columns.to_list(), sentence_df.columns.to_list())
        self.assertNotIn("index", saved_df.columns)
        self.assertEqual(saved_df.loc[0, "polygon_name"], "Bois Alpha")
        self.assertEqual(
            saved_df.loc[0, "sentence"],
            (
                "This forest contains mixed canopy and wetland habitat "
                "visible from satellite imagery."
            ),
        )


if __name__ == "__main__":
    unittest.main()

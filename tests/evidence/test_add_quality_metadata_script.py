import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import scripts.evidence.add_quality_metadata as add_quality_metadata_script


class AddQualityMetadataScriptTests(unittest.TestCase):
    def test_builds_quality_metadata_artifact(self):
        page_text_df = pd.DataFrame(
            [
                {
                    "url": "https://example.com/a",
                    "text": "This forest has mixed canopy.\nWetlands border the river.",
                },
                {
                    "url": "https://example.com/b",
                    "text": None,
                },
            ],
            index=[10, 11],
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "input.parquet"
            output_path = temp_path / "nested" / "quality.parquet"
            page_text_df.to_parquet(input_path, index=False)

            self.assertTrue(
                hasattr(add_quality_metadata_script, "run_quality_metadata_build")
            )
            result = add_quality_metadata_script.run_quality_metadata_build(
                input_path=input_path,
                output_path=output_path,
            )

            saved_df = pd.read_parquet(output_path)

        self.assertEqual(len(result), 2)
        self.assertEqual(saved_df.columns.to_list(), result.columns.to_list())
        self.assertIn("quality_score", saved_df.columns)
        self.assertEqual(saved_df.loc[1, "quality_flags"], ["empty_text"])
        self.assertEqual(saved_df.loc[1, "quality_score"], 0.0)


if __name__ == "__main__":
    unittest.main()

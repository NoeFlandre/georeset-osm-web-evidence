import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from openpyxl import load_workbook

from georeset_osm_web_evidence.review.artifacts import (
    format_review_artifact_summary,
    write_review_artifacts,
)


class ReviewArtifactTests(unittest.TestCase):
    def test_writes_review_csv_and_xlsx_with_supplied_workbook_writer(self):
        review_df = pd.DataFrame(
            [
                {
                    "review_id": "review-0001",
                    "human_label": "",
                    "polygon_name": "Forest A",
                }
            ]
        )

        def write_workbook(dataframe: pd.DataFrame, path: str | Path) -> None:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                dataframe.to_excel(writer, sheet_name="review", index=False)

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            csv_output_path = temp_path / "nested" / "review.csv"
            xlsx_output_path = temp_path / "nested" / "review.xlsx"

            result_df = write_review_artifacts(
                review_df,
                csv_output_path=csv_output_path,
                xlsx_output_path=xlsx_output_path,
                workbook_writer=write_workbook,
            )

            saved_csv_df = pd.read_csv(csv_output_path)
            workbook = load_workbook(xlsx_output_path)

        self.assertIs(result_df, review_df)
        self.assertEqual(saved_csv_df.loc[0, "review_id"], "review-0001")
        self.assertEqual(workbook["review"]["A1"].value, "review_id")

    def test_formats_review_artifact_summary(self):
        review_df = pd.DataFrame([{"review_id": "review-0001"}])

        summary = format_review_artifact_summary(
            review_df,
            csv_output_path=Path("review.csv"),
            xlsx_output_path=Path("review.xlsx"),
        )

        self.assertEqual(
            summary,
            "Saved 1 review rows to review.csv\n"
            "Saved reviewer workbook to review.xlsx",
        )


if __name__ == "__main__":
    unittest.main()

from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.review.artifacts import write_review_artifacts
from georeset_osm_web_evidence.review.human import (
    build_human_review_dataframe,
    save_human_review_xlsx,
)


def main() -> None:
    input_path = "data/processed/evidence/page_text_sample.parquet"
    csv_output_path = Path("data/review/trafilatura/human_review_sample.csv")
    xlsx_output_path = Path("data/review/trafilatura/human_review_sample.xlsx")

    page_text_df = pd.read_parquet(input_path)
    review_df = build_human_review_dataframe(page_text_df)

    write_review_artifacts(
        review_df,
        csv_output_path=csv_output_path,
        xlsx_output_path=xlsx_output_path,
        workbook_writer=save_human_review_xlsx,
    )

    print(f"Saved {len(review_df)} review rows to {csv_output_path}")
    print(f"Saved reviewer workbook to {xlsx_output_path}")
    print(review_df[["review_id", "human_label", "fetch_status", "polygon_name"]])


if __name__ == "__main__":
    main()

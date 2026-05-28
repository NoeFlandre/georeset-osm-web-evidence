from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.review.human import build_human_review_dataframe


def main() -> None:
    input_path = "data/processed/evidence/page_text_sample.parquet"
    output_path = "data/review/human_review_sample.csv"

    page_text_df = pd.read_parquet(input_path)
    review_df = build_human_review_dataframe(page_text_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_path, index=False)

    print(f"Saved {len(review_df)} review rows to {output_path}")
    print(review_df[["review_id", "human_label", "fetch_status", "polygon_name"]])


if __name__ == "__main__":
    main()

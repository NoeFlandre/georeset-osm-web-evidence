from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.review.sentence_labels import (
    build_sentence_label_review_dataframe,
    save_sentence_label_review_xlsx,
)

DEFAULT_INPUT_PATH = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_context_queries_v1/"
    "sentence_candidates_llm_labeled.parquet"
)
DEFAULT_CSV_OUTPUT_PATH = Path(
    "data/review/english_sentence_pilot_context_queries_v1/"
    "llm_labeled_english_sentence_pilot_context_queries_v1.csv"
)
DEFAULT_XLSX_OUTPUT_PATH = Path(
    "data/review/english_sentence_pilot_context_queries_v1/"
    "llm_labeled_english_sentence_pilot_context_queries_v1.xlsx"
)


def run_context_query_sentence_review_build(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    csv_output_path: str | Path = DEFAULT_CSV_OUTPUT_PATH,
    xlsx_output_path: str | Path = DEFAULT_XLSX_OUTPUT_PATH,
) -> pd.DataFrame:
    input_path = Path(input_path)
    csv_output_path = Path(csv_output_path)
    xlsx_output_path = Path(xlsx_output_path)

    labeled_df = pd.read_parquet(input_path)
    review_df = build_sentence_label_review_dataframe(labeled_df)

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(csv_output_path, index=False)
    save_sentence_label_review_xlsx(review_df, xlsx_output_path)

    return review_df


def main() -> None:
    review_df = run_context_query_sentence_review_build()

    print(f"Saved {len(review_df)} review rows to {DEFAULT_CSV_OUTPUT_PATH}")
    print(f"Saved reviewer workbook to {DEFAULT_XLSX_OUTPUT_PATH}")
    print(review_df[["review_id", "human_label", "llm_label", "polygon_name"]])


if __name__ == "__main__":
    main()

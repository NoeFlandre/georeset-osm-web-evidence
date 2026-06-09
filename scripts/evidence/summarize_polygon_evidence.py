from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.summary import summarize_polygon_evidence
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact

DEFAULT_POLYGONS_INPUT_PATH = Path(
    "data/processed/samples/balanced_wikipedia_100.parquet"
)
DEFAULT_PAGE_TEXT_INPUT_PATH = Path(
    "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
)
DEFAULT_OUTPUT_PATH = Path("data/processed/evidence/polygon_evidence_summary.parquet")


def run_polygon_evidence_summary_build(
    polygons_df_input_path: str | Path = DEFAULT_POLYGONS_INPUT_PATH,
    page_text_df_input_path: str | Path = DEFAULT_PAGE_TEXT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    high_quality_threshold: float = 0.8,
) -> pd.DataFrame:
    polygons_df = pd.read_parquet(polygons_df_input_path)
    page_text_df = pd.read_parquet(page_text_df_input_path)

    summary_df = summarize_polygon_evidence(
        polygons_df,
        page_text_df,
        high_quality_threshold=high_quality_threshold,
    )
    return write_dataframe_artifact(summary_df, output_path)


def main():
    run_polygon_evidence_summary_build(
        polygons_df_input_path=DEFAULT_POLYGONS_INPUT_PATH,
        page_text_df_input_path=DEFAULT_PAGE_TEXT_INPUT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
    )
    print(f"Saved the summary to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

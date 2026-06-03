from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.summary import summarize_polygon_evidence


def main():
    polygons_df_input_path = "data/processed/samples/balanced_wikipedia_100.parquet"
    page_text_df_input_path = (
        "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
    )
    output_path = Path("data/processed/evidence/polygon_evidence_summary.parquet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    polygons_df = pd.read_parquet(polygons_df_input_path)
    page_text_df = pd.read_parquet(page_text_df_input_path)

    summary_df = summarize_polygon_evidence(polygons_df, page_text_df)
    summary_df.to_parquet(output_path, index=False)
    print(f"Saved the summary to {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd


def main():
    polygon_summary_path = Path(
        "data/processed/evidence/polygon_evidence_summary.parquet"
    )
    page_quality_path = Path(
        "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
    )

    polygon_summary_df = pd.read_parquet(polygon_summary_path)
    page_quality_df = pd.read_parquet(page_quality_path)

    print("========Polygon-level========")
    print(f"Number of polygons: {len(polygon_summary_df)}")
    print(
        f"Polygons with high quality evidence: {polygon_summary_df['has_high_quality_evidence'].value_counts(dropna=False)}"
    )

    print("===== Evidence by Wikipedia status=====")
    print(
        pd.crosstab(
            polygon_summary_df["has_wikipedia_articles"],
            polygon_summary_df["has_high_quality_evidence"],
            margins=True,
        )
    )

    print("=====Page-level=======")
    print(f"Number of candidate URLS: {len(page_quality_df)}")
    print(f"Page level stats: {page_quality_df['quality_score'].describe()}")

    print("=====Quality Flags====")
    print(page_quality_df["quality_flags"].astype(str).value_counts(dropna=False))


if __name__ == "__main__":
    main()

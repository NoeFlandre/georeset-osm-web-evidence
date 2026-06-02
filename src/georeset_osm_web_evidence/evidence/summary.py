import pandas as pd

POLYGON_KEY = ["osm_type", "osm_id"]


def summarize_polygon_evidence(
    polygons_df: pd.DataFrame,
    page_text_df: pd.DataFrame,
    high_quality_threshold: float = 0.8,
) -> pd.DataFrame:
    page_text_df = page_text_df.copy()
    page_text_df["is_successful_fetch"] = page_text_df["fetch_error"].isna()
    page_text_df["is_high_quality"] = (
        page_text_df["quality_score"] >= high_quality_threshold
    )

    page_summary_df = page_text_df.groupby(POLYGON_KEY, as_index=False).agg(
        candidate_url_count=("source_url", "nunique"),
        successful_fetch_count=("is_successful_fetch", "sum"),
        high_quality_page_count=("is_high_quality", "sum"),
        mean_quality_score=("quality_score", "mean"),
        max_quality_score=("quality_score", "max"),
    )

    summary_df = polygons_df.merge(
        page_summary_df,
        on=POLYGON_KEY,
        how="left",
    )

    count_columns = [
        "candidate_url_count",
        "successful_fetch_count",
        "high_quality_page_count",
    ]
    score_columns = ["mean_quality_score", "max_quality_score"]

    for column in count_columns:
        summary_df[column] = summary_df[column].fillna(0).astype(int)

    for column in score_columns:
        summary_df[column] = summary_df[column].fillna(0.0)

    summary_df["has_high_quality_evidence"] = summary_df["high_quality_page_count"] > 0

    return summary_df

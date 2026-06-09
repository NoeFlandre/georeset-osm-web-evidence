import pandas as pd


POLYGON_KEY_COLUMNS = ["osm_type", "osm_id"]


def _sentence_url_counts(sentence_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sentence_df.groupby(POLYGON_KEY_COLUMNS, dropna=False)
        .agg(sentence_count=("sentence", "size"), url_count=("url", "nunique"))
        .reset_index()
    )


def validate_exact_sentence_url_counts(
    sentence_df: pd.DataFrame,
    urls_per_polygon: int,
) -> None:
    counts_df = _sentence_url_counts(sentence_df)
    invalid_counts_df = counts_df[
        (counts_df["sentence_count"] != urls_per_polygon)
        | (counts_df["url_count"] != urls_per_polygon)
    ]

    if not invalid_counts_df.empty:
        raise ValueError(
            "Selected sentence artifact does not have exactly "
            f"{urls_per_polygon} sentences and URLs per polygon: "
            f"{invalid_counts_df.to_dict('records')}"
        )


def select_exact_url_artifacts(
    sentence_df: pd.DataFrame,
    candidate_urls_df: pd.DataFrame,
    page_text_df: pd.DataFrame,
    urls_per_polygon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_exact_sentence_url_counts(sentence_df, urls_per_polygon=urls_per_polygon)

    sentence_urls_df = sentence_df[POLYGON_KEY_COLUMNS + ["url"]].drop_duplicates()
    selected_candidate_urls_df = sentence_urls_df.merge(
        candidate_urls_df,
        on=POLYGON_KEY_COLUMNS + ["url"],
        how="left",
    )
    page_text_keys_df = sentence_urls_df.rename(columns={"url": "source_url"})
    selected_page_text_df = page_text_keys_df.merge(
        page_text_df,
        on=POLYGON_KEY_COLUMNS + ["source_url"],
        how="left",
    )

    if selected_candidate_urls_df["best_rank"].isna().any():
        raise ValueError("Some selected sentence URLs are missing from candidate URLs")
    if selected_page_text_df["source_url"].isna().any():
        raise ValueError("Some selected sentence URLs are missing from page text")

    return selected_candidate_urls_df, selected_page_text_df

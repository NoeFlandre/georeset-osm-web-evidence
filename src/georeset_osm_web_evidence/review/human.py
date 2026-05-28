import pandas as pd


REVIEW_COLUMNS = [
    "review_id",
    "human_label",
    "human_notes",
    "fetch_status",
    "polygon_name",
    "source_url",
    "search_title",
    "search_description",
    "page_title",
    "text_preview",
    "fetch_error",
    "text_length",
    "osm_type",
    "osm_id",
    "has_wikipedia_articles",
]


def clean_text_for_review(text: str | None) -> str:
    if not isinstance(text, str):
        return ""

    return " ".join(text.split())


def make_text_preview(text: str | None, preview_chars: int = 1500) -> str:
    clean_text = clean_text_for_review(text)

    if len(clean_text) <= preview_chars:
        return clean_text

    return clean_text[:preview_chars].rstrip() + "…"


def select_successful_review_rows(
    page_text_df: pd.DataFrame,
    max_rows: int = 30,
    max_rows_per_polygon: int = 2,
) -> pd.DataFrame:
    successful_df = page_text_df[
        page_text_df["fetch_error"].isna()
        & page_text_df["text"].apply(lambda text: isinstance(text, str) and bool(text))
    ].copy()

    successful_df = successful_df.sort_values(
        [
            "has_wikipedia_articles",
            "osm_type",
            "polygon_name",
            "source_url",
        ]
    )

    capped_df = successful_df.groupby("polygon_name", group_keys=False).head(
        max_rows_per_polygon
    )

    return capped_df.head(max_rows).reset_index(drop=True)


def build_human_review_dataframe(
    page_text_df: pd.DataFrame,
    preview_chars: int = 1500,
    max_rows: int = 30,
    max_rows_per_polygon: int = 2,
) -> pd.DataFrame:
    review_df = select_successful_review_rows(
        page_text_df,
        max_rows=max_rows,
        max_rows_per_polygon=max_rows_per_polygon,
    )

    review_df = review_df.sort_values(["polygon_name", "source_url"]).reset_index(
        drop=True
    )
    review_df["review_id"] = [
        f"review-{index:04d}" for index in range(1, len(review_df) + 1)
    ]
    review_df["human_label"] = ""
    review_df["human_notes"] = ""
    review_df["fetch_status"] = review_df["fetch_error"].apply(
        lambda value: "broken" if isinstance(value, str) and value else "fetched"
    )
    review_df["page_title"] = review_df["title"].fillna("")
    review_df["text_preview"] = review_df["text"].apply(
        lambda text: make_text_preview(text, preview_chars=preview_chars)
    )
    review_df["fetch_error"] = review_df["fetch_error"].fillna("")

    return review_df[REVIEW_COLUMNS]

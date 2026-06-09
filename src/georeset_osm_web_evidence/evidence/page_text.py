import logging
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import pandas as pd

from georeset_osm_web_evidence.web.text import fetch_page_text


PAGE_TEXT_COLUMNS = [
    "osm_type",
    "osm_id",
    "polygon_name",
    "has_wikipedia_articles",
    "provider",
    "source_url",
    "search_title",
    "search_description",
    "search_queries",
    "url",
    "final_url",
    "status_code",
    "title",
    "text",
    "text_length",
    "fetch_error",
    "extraction_method",
    "extraction_error",
    "best_rank",
    "world_region",
    "country",
    "local_language",
    "query_local_language",
    "query_language",
    "area_size_bin",
    "polygon_category",
]
PAGE_TEXT_CACHE_METADATA_COLUMNS = [
    "osm_type",
    "osm_id",
    "polygon_name",
    "has_wikipedia_articles",
    "provider",
    "search_title",
    "search_description",
    "search_queries",
    "best_rank",
    "world_region",
    "country",
    "local_language",
    "query_local_language",
    "query_language",
    "area_size_bin",
    "polygon_category",
]


def combine_queries_for_review(queries) -> str:
    if queries is None:
        return ""

    return "; ".join(str(query) for query in queries)


def build_page_text_row(candidate_url_row, page_text: dict) -> dict:
    return {
        "osm_type": candidate_url_row.osm_type,
        "osm_id": candidate_url_row.osm_id,
        "polygon_name": candidate_url_row.polygon_name,
        "has_wikipedia_articles": candidate_url_row.has_wikipedia_articles,
        "provider": candidate_url_row.provider,
        "source_url": candidate_url_row.url,
        "search_title": candidate_url_row.title,
        "search_description": candidate_url_row.description,
        "search_queries": combine_queries_for_review(candidate_url_row.queries),
        **page_text,
    }


def page_text_metadata_value(candidate_url_row: pd.Series, column: str):
    if column == "search_title":
        return candidate_url_row["title"]
    if column == "search_description":
        return candidate_url_row["description"]
    if column == "search_queries":
        return combine_queries_for_review(candidate_url_row["queries"])

    return candidate_url_row.get(column, pd.NA)


def backfill_cached_page_text_metadata(
    page_text_df: pd.DataFrame,
    candidate_urls_df: pd.DataFrame,
    metadata_columns: list[str],
) -> tuple[pd.DataFrame, bool]:
    if page_text_df.empty:
        return page_text_df.copy(), False

    candidate_lookup_df = candidate_urls_df.drop_duplicates("url").set_index("url")
    result_df = page_text_df.copy()
    changed = False

    for row_index, source_url in result_df["source_url"].items():
        if source_url not in candidate_lookup_df.index:
            continue

        candidate_url_row = candidate_lookup_df.loc[source_url]
        for column in metadata_columns:
            new_value = page_text_metadata_value(candidate_url_row, column)
            current_value = result_df.at[row_index, column]
            if pd.isna(current_value) and not pd.isna(new_value):
                result_df.at[row_index, column] = new_value
                changed = True

    return result_df, changed


def page_text_quality_artifact_is_usable(
    page_text_with_quality_df: pd.DataFrame,
) -> bool:
    required_columns = ["source_url", "quality_score", "query_language"]
    if any(column not in page_text_with_quality_df.columns for column in required_columns):
        return False
    if page_text_with_quality_df.empty:
        return False

    return not page_text_with_quality_df["query_language"].isna().any()


def is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def skipped_page_text_result(url: str, fetch_error: str) -> dict:
    return {
        "url": url,
        "final_url": None,
        "status_code": None,
        "title": None,
        "text": None,
        "text_length": 0,
        "fetch_error": fetch_error,
        "extraction_method": None,
        "extraction_error": None,
    }


def fetch_candidate_pages(
    candidate_urls_df: pd.DataFrame,
    logger: logging.Logger | None,
    output_path: Path,
    reset: bool = False,
    stop_when: Callable[[pd.DataFrame], bool] | None = None,
    stop_check_interval: int = 10,
    fetch_timeout_seconds: int = 10,
    fetch_delay_seconds: float = 1.0,
    fetch_page_text_func: Callable[..., dict] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, bool]:
    if stop_check_interval <= 0:
        raise ValueError("stop_check_interval must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not reset:
        page_text_df = pd.read_parquet(output_path)
        if "source_url" not in page_text_df.columns:
            raise ValueError(f"{output_path} is missing required column: source_url")
        for column in PAGE_TEXT_COLUMNS:
            if column not in page_text_df.columns:
                page_text_df[column] = pd.NA
        page_text_df = page_text_df[PAGE_TEXT_COLUMNS]
    else:
        page_text_df = pd.DataFrame(columns=PAGE_TEXT_COLUMNS)

    candidate_url_set = set(candidate_urls_df["url"])
    cached_row_count = len(page_text_df)
    page_text_df = page_text_df[
        page_text_df["source_url"].isin(candidate_url_set)
    ].reset_index(drop=True)
    changed = len(page_text_df) != cached_row_count

    page_text_df, metadata_changed = backfill_cached_page_text_metadata(
        page_text_df,
        candidate_urls_df,
        metadata_columns=PAGE_TEXT_CACHE_METADATA_COLUMNS,
    )
    changed = changed or metadata_changed
    if changed:
        page_text_df.to_parquet(output_path, index=False)

    fetched_urls = set(page_text_df["source_url"].dropna())
    if stop_when is not None and stop_when(page_text_df):
        if logger is not None:
            logger.info("Page fetch quota is already satisfied from cached rows")
        return page_text_df, changed

    page_text_fetcher = fetch_page_text_func or fetch_page_text
    for url_index, row in enumerate(candidate_urls_df.itertuples(), start=1):
        if row.url in fetched_urls:
            if logger is not None:
                logger.info(
                    "Skipping already fetched URL %s/%s: %s",
                    url_index,
                    len(candidate_urls_df),
                    row.url,
                )
            continue

        if is_pdf_url(row.url):
            if logger is not None:
                logger.info(
                    "Skipping PDF URL %s/%s: %s",
                    url_index,
                    len(candidate_urls_df),
                    row.url,
                )
            page_text = skipped_page_text_result(row.url, "Skipped PDF URL")
        else:
            if logger is not None:
                logger.info(
                    "Fetching URL %s/%s: %s",
                    url_index,
                    len(candidate_urls_df),
                    row.url,
                )
            page_text = page_text_fetcher(
                row.url,
                timeout_seconds=fetch_timeout_seconds,
            )
        page_row = build_page_text_row(row, page_text)
        page_row.update(
            {
                "best_rank": row.best_rank,
                "world_region": row.world_region,
                "country": row.country,
                "local_language": row.local_language,
                "query_local_language": row.query_local_language,
                "query_language": getattr(row, "query_language", None),
                "area_size_bin": row.area_size_bin,
                "polygon_category": row.polygon_category,
            }
        )
        page_text_df.loc[len(page_text_df)] = [
            page_row.get(column) for column in PAGE_TEXT_COLUMNS
        ]
        page_text_df.to_parquet(output_path, index=False)
        fetched_urls.add(row.url)
        changed = True

        if (
            stop_when is not None
            and len(page_text_df) % stop_check_interval == 0
            and stop_when(page_text_df)
        ):
            if logger is not None:
                logger.info("Stopping page fetch because sentence quota is satisfied")
            break

        sleep_func(fetch_delay_seconds)

    return page_text_df, changed

from datetime import datetime, timezone

import pandas as pd

from georeset_osm_web_evidence.storage.dataframe import append_unique_rows

SEARCH_RESULT_UNIQUE_KEY = ["osm_type", "osm_id", "query", "url"]
SEARCH_ATTEMPT_UNIQUE_KEY = ["osm_type", "osm_id", "query"]


def is_wikipedia_url(url: str) -> bool:
    return "wikipedia.org" in url.lower()


def combine_unique_values(values) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def result_to_row(
    polygon_row,
    polygon_name: str,
    query: str,
    rank: int,
    result: dict,
) -> dict:
    return {
        "osm_type": polygon_row.osm_type,
        "osm_id": polygon_row.osm_id,
        "polygon_name": polygon_name,
        "has_wikipedia_articles": polygon_row.has_wikipedia_articles,
        "query": query,
        "provider": result["provider"],
        "rank": rank,
        "title": result["title"],
        "url": result["url"],
        "description": result["description"],
    }


def attempt_to_row(
    polygon_row,
    polygon_name: str,
    query: str,
    result_count: int,
) -> dict:
    return {
        "osm_type": polygon_row.osm_type,
        "osm_id": polygon_row.osm_id,
        "polygon_name": polygon_name,
        "has_wikipedia_articles": polygon_row.has_wikipedia_articles,
        "query": query,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "result_count": result_count,
    }


def prepare_candidate_urls(search_results_df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = search_results_df[
        ~search_results_df["url"].apply(is_wikipedia_url)
    ].copy()

    return (
        filtered_df.sort_values("rank")
        .groupby(
            [
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "provider",
                "url",
            ],
            as_index=False,
        )
        .agg(
            best_rank=("rank", "min"),
            title=("title", "first"),
            description=("description", "first"),
            queries=("query", combine_unique_values),
        )
    )


def merge_search_results(
    existing_results_df: pd.DataFrame,
    new_results_df: pd.DataFrame,
) -> pd.DataFrame:
    return append_unique_rows(
        existing_results_df,
        new_results_df,
        subset=SEARCH_RESULT_UNIQUE_KEY,
    )


def merge_search_attempts(
    existing_attempts_df: pd.DataFrame,
    new_attempts_df: pd.DataFrame,
) -> pd.DataFrame:
    return append_unique_rows(
        existing_attempts_df,
        new_attempts_df,
        subset=SEARCH_ATTEMPT_UNIQUE_KEY,
    )

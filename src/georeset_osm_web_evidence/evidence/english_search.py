import logging
import time
from typing import Callable

import pandas as pd

from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    build_search_rows_for_query,
)
from georeset_osm_web_evidence.search.providers import search_brave


SEARCH_RESULT_COLUMNS = [
    "osm_type",
    "osm_id",
    "polygon_name",
    "has_wikipedia_articles",
    "query",
    "provider",
    "rank",
    "title",
    "url",
    "description",
    "query_language",
    "world_region",
    "country",
    "local_language",
    "query_local_language",
    "area_size_bin",
    "polygon_category",
]
SEARCH_ATTEMPT_COLUMNS = [
    "osm_type",
    "osm_id",
    "polygon_name",
    "has_wikipedia_articles",
    "query",
    "attempted_at",
    "result_count",
    "query_language",
    "search_error",
]


def empty_english_search_results() -> pd.DataFrame:
    return pd.DataFrame(columns=SEARCH_RESULT_COLUMNS)


def empty_english_search_attempts() -> pd.DataFrame:
    return pd.DataFrame(columns=SEARCH_ATTEMPT_COLUMNS)


def search_english_queries_for_polygon(
    polygon_row,
    queries: list[str],
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    results_per_query: int = 20,
    request_delay_seconds: float = 1.2,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    attempt_rows = []

    for query in queries:
        search_error = None
        try:
            results = search_func(
                query,
                count=results_per_query,
                country="US",
                search_lang="en",
            )
        except Exception as error:
            results = []
            search_error = str(error)
            if logger is not None:
                logger.warning("Search failed for %s: %s", query, search_error)

        result_rows, attempt_row = build_search_rows_for_query(
            polygon_row=polygon_row,
            query_language="en",
            query=query,
            results=results,
            search_error=search_error,
        )
        rows.extend(result_rows)
        attempt_rows.append(attempt_row)
        sleep_func(request_delay_seconds)

    search_results_df = (
        pd.DataFrame(rows, columns=SEARCH_RESULT_COLUMNS)
        if not rows
        else pd.DataFrame(rows)
    )
    search_attempts_df = (
        pd.DataFrame(attempt_rows, columns=SEARCH_ATTEMPT_COLUMNS)
        if not attempt_rows
        else pd.DataFrame(attempt_rows)
    )

    return search_results_df, search_attempts_df


def build_english_search_artifacts(
    pilot_gdf: pd.DataFrame,
    query_builder: Callable[[object, int], list[str]],
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    max_queries_per_polygon: int = 4,
    results_per_query: int = 20,
    request_delay_seconds: float = 1.2,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    search_result_parts = []
    search_attempt_parts = []

    for polygon_index, polygon_row in enumerate(pilot_gdf.itertuples(), start=1):
        if logger is not None:
            logger.info(
                "Searching polygon %s/%s: %s",
                polygon_index,
                len(pilot_gdf),
                polygon_row.polygon_name,
            )

        queries = query_builder(polygon_row, max_queries_per_polygon)
        search_results_df, search_attempts_df = search_english_queries_for_polygon(
            polygon_row,
            queries=queries,
            search_func=search_func,
            sleep_func=sleep_func,
            results_per_query=results_per_query,
            request_delay_seconds=request_delay_seconds,
            logger=logger,
        )
        search_result_parts.append(search_results_df)
        search_attempt_parts.append(search_attempts_df)

    search_results_df = (
        pd.concat(search_result_parts, ignore_index=True)
        if search_result_parts
        else empty_english_search_results()
    )
    search_attempts_df = (
        pd.concat(search_attempt_parts, ignore_index=True)
        if search_attempt_parts
        else empty_english_search_attempts()
    )

    return search_results_df, search_attempts_df

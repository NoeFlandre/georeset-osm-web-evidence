import logging
import time
from typing import Callable

import pandas as pd

from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    build_search_rows_for_query,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import (
    build_location_topic_english_search_queries,
)


def search_location_topic_for_polygon(
    polygon_row,
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    max_queries_per_polygon: int = 4,
    results_per_query: int = 20,
    request_delay_seconds: float = 1.2,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = build_location_topic_english_search_queries(
        osm_tags=polygon_row.osm_tags,
        country=polygon_row.country,
        world_region=polygon_row.world_region,
        source_extract_id=polygon_row.source_extract_id,
        polygon_category=polygon_row.polygon_category,
        max_queries=max_queries_per_polygon,
    )
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

    return pd.DataFrame(rows), pd.DataFrame(attempt_rows)


def build_location_topic_search_artifacts(
    pilot_gdf: pd.DataFrame,
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

        search_results_df, search_attempts_df = search_location_topic_for_polygon(
            polygon_row,
            search_func=search_func,
            sleep_func=sleep_func,
            max_queries_per_polygon=max_queries_per_polygon,
            results_per_query=results_per_query,
            request_delay_seconds=request_delay_seconds,
            logger=logger,
        )
        search_result_parts.append(search_results_df)
        search_attempt_parts.append(search_attempts_df)

    search_results_df = (
        pd.concat(search_result_parts, ignore_index=True)
        if search_result_parts
        else pd.DataFrame()
    )
    search_attempts_df = (
        pd.concat(search_attempt_parts, ignore_index=True)
        if search_attempt_parts
        else pd.DataFrame()
    )

    return search_results_df, search_attempts_df

import logging
import time
from typing import Callable

import pandas as pd

from georeset_osm_web_evidence.evidence.english_search import (
    build_english_search_artifacts,
    empty_english_search_attempts,
    empty_english_search_results,
    search_english_queries_for_polygon,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import (
    build_contextual_english_search_queries,
)


def _context_queries_for_polygon(
    polygon_row,
    max_queries_per_polygon: int,
) -> list[str]:
    return build_contextual_english_search_queries(
        osm_tags=polygon_row.osm_tags,
        country=polygon_row.country,
        world_region=polygon_row.world_region,
        source_extract_id=polygon_row.source_extract_id,
        polygon_category=polygon_row.polygon_category,
        max_queries=max_queries_per_polygon,
    )


def empty_context_query_search_results() -> pd.DataFrame:
    return empty_english_search_results()


def empty_context_query_search_attempts() -> pd.DataFrame:
    return empty_english_search_attempts()


def search_context_queries_for_polygon(
    polygon_row,
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    max_queries_per_polygon: int = 4,
    results_per_query: int = 20,
    request_delay_seconds: float = 1.2,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = _context_queries_for_polygon(polygon_row, max_queries_per_polygon)
    return search_english_queries_for_polygon(
        polygon_row,
        queries=queries,
        search_func=search_func,
        sleep_func=sleep_func,
        results_per_query=results_per_query,
        request_delay_seconds=request_delay_seconds,
        logger=logger,
    )


def build_context_query_search_artifacts(
    pilot_gdf: pd.DataFrame,
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    max_queries_per_polygon: int = 4,
    results_per_query: int = 20,
    request_delay_seconds: float = 1.2,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_english_search_artifacts(
        pilot_gdf,
        query_builder=_context_queries_for_polygon,
        search_func=search_func,
        sleep_func=sleep_func,
        max_queries_per_polygon=max_queries_per_polygon,
        results_per_query=results_per_query,
        request_delay_seconds=request_delay_seconds,
        logger=logger,
    )

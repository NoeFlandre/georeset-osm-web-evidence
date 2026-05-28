import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.coverage import (
    build_expected_query_table,
    choose_unsearched_polygons,
    find_missing_queries,
    load_existing_search_attempts,
    load_existing_search_results,
)
from georeset_osm_web_evidence.search.queries import get_osm_name
from georeset_osm_web_evidence.storage.local import load_geodataframe


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


def choose_polygons_to_search(
    polygons_df: pd.DataFrame,
    existing_results_df: pd.DataFrame,
    existing_attempts_df: pd.DataFrame,
    new_polygon_limit: int,
    complete_existing_polygons_only: bool,
) -> pd.DataFrame:
    if not complete_existing_polygons_only:
        return choose_unsearched_polygons(
            polygons_df,
            existing_results_df,
            polygon_limit=new_polygon_limit,
            attempted_polygons_df=existing_attempts_df,
        )

    treated_keys = pd.concat(
        [
            existing_results_df[["osm_type", "osm_id"]],
            existing_attempts_df[["osm_type", "osm_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    return polygons_df.merge(treated_keys, on=["osm_type", "osm_id"], how="inner")


def main() -> None:
    input_path = "data/processed/samples/balanced_wikipedia_100.parquet"
    output_path = "data/processed/search/brave_results_sample.parquet"
    attempts_path = "data/processed/search/brave_search_attempts.parquet"
    new_polygon_limit = 10
    complete_existing_polygons_only = True
    results_per_query = 5
    request_delay_seconds = 1.2

    gdf = load_geodataframe(input_path)
    existing_results_df = load_existing_search_results(output_path)
    existing_attempts_df = load_existing_search_attempts(attempts_path)
    gdf = choose_polygons_to_search(
        gdf,
        existing_results_df,
        existing_attempts_df,
        new_polygon_limit=new_polygon_limit,
        complete_existing_polygons_only=complete_existing_polygons_only,
    )
    missing_queries_df = find_missing_queries(
        build_expected_query_table(gdf),
        existing_results_df,
        attempted_queries_df=existing_attempts_df,
    )
    rows = []
    attempt_rows = []

    for polygon_index, polygon_row in enumerate(gdf.itertuples(), start=1):
        polygon_name = get_osm_name(polygon_row.osm_tags)
        polygon_missing_queries = missing_queries_df[
            (missing_queries_df["osm_type"] == polygon_row.osm_type)
            & (missing_queries_df["osm_id"] == polygon_row.osm_id)
        ]["query"].to_list()

        if not polygon_missing_queries:
            continue

        print(
            f"Searching polygon {polygon_index}/{len(gdf)}: "
            f"{polygon_name} ({len(polygon_missing_queries)} missing queries)"
        )

        for query in polygon_missing_queries:
            results = search_brave(query, count=results_per_query)
            attempt_rows.append(
                attempt_to_row(
                    polygon_row=polygon_row,
                    polygon_name=polygon_name,
                    query=query,
                    result_count=len(results),
                )
            )

            for rank, result in enumerate(results, start=1):
                rows.append(
                    result_to_row(
                        polygon_row=polygon_row,
                        polygon_name=polygon_name,
                        query=query,
                        rank=rank,
                        result=result,
                    )
                )

            time.sleep(request_delay_seconds)

    output_path = Path(output_path)
    attempts_path = Path(attempts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_path.parent.mkdir(parents=True, exist_ok=True)

    new_results_df = pd.DataFrame(rows, columns=existing_results_df.columns)
    results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
    results_df = results_df.drop_duplicates(
        subset=["osm_type", "osm_id", "query", "url"],
        keep="first",
    )
    results_df.to_parquet(output_path, index=False)

    new_attempts_df = pd.DataFrame(attempt_rows, columns=existing_attempts_df.columns)
    attempts_df = pd.concat(
        [existing_attempts_df, new_attempts_df],
        ignore_index=True,
    )
    attempts_df = attempts_df.drop_duplicates(
        subset=["osm_type", "osm_id", "query"],
        keep="first",
    )
    attempts_df.to_parquet(attempts_path, index=False)

    print(f"Collected {len(new_results_df)} new search results")
    print(f"Saved {len(results_df)} total search results to {output_path}")
    print(f"Saved {len(attempts_df)} attempted polygons to {attempts_path}")


if __name__ == "__main__":
    main()

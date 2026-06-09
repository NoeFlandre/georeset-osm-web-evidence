import time

import pandas as pd

from georeset_osm_web_evidence.search.config import (
    BALANCED_POLYGONS_PATH,
    BRAVE_ATTEMPTS_PATH,
    BRAVE_RESULTS_PATH,
    SEARCH_LANGUAGES,
)
from georeset_osm_web_evidence.search.coverage import (
    build_expected_query_table,
    choose_polygons_to_search,
    find_missing_queries,
    load_existing_search_attempts,
    load_existing_search_results,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import get_osm_name
from georeset_osm_web_evidence.search.results import (
    attempt_to_row,
    merge_search_attempts,
    merge_search_results,
    result_to_row,
)
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact
from georeset_osm_web_evidence.storage.local import load_geodataframe


def main() -> None:
    new_polygon_limit = 81
    complete_existing_polygons_only = False
    results_per_query = 5
    request_delay_seconds = 1.2

    gdf = load_geodataframe(BALANCED_POLYGONS_PATH)

    existing_results_df = load_existing_search_results(BRAVE_RESULTS_PATH)
    existing_attempts_df = load_existing_search_attempts(BRAVE_ATTEMPTS_PATH)
    gdf = choose_polygons_to_search(
        gdf,
        existing_results_df,
        existing_attempts_df,
        new_polygon_limit=new_polygon_limit,
        complete_existing_polygons_only=complete_existing_polygons_only,
    )

    if gdf.empty:
        print("No unsearched polygons left")
        return

    missing_queries_df = find_missing_queries(
        build_expected_query_table(gdf, search_languages=SEARCH_LANGUAGES),
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

    output_path = BRAVE_RESULTS_PATH
    attempts_path = BRAVE_ATTEMPTS_PATH

    new_results_df = pd.DataFrame(rows, columns=existing_results_df.columns)
    results_df = merge_search_results(existing_results_df, new_results_df)
    write_dataframe_artifact(results_df, output_path)

    new_attempts_df = pd.DataFrame(attempt_rows, columns=existing_attempts_df.columns)
    attempts_df = merge_search_attempts(existing_attempts_df, new_attempts_df)
    write_dataframe_artifact(attempts_df, attempts_path)

    print(f"Collected {len(new_results_df)} new search results")
    print(f"Saved {len(results_df)} total search results to {output_path}")
    print(f"Saved {len(attempts_df)} attempted polygons to {attempts_path}")


if __name__ == "__main__":
    main()

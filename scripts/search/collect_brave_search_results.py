import time
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import build_search_queries, get_osm_name
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


def main() -> None:
    input_path = "data/processed/samples/balanced_wikipedia_100.parquet"
    output_path = "data/processed/search/brave_results_sample.parquet"
    polygon_limit = 5
    results_per_query = 5
    request_delay_seconds = 1.2

    gdf = load_geodataframe(input_path).head(polygon_limit)
    rows = []

    for polygon_index, polygon_row in enumerate(gdf.itertuples(), start=1):
        polygon_name = get_osm_name(polygon_row.osm_tags)
        queries = build_search_queries(polygon_row.osm_tags)

        print(f"Searching polygon {polygon_index}/{len(gdf)}: {polygon_name}")

        for query in queries:
            results = search_brave(query, count=results_per_query)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(rows)
    results_df.to_parquet(output_path, index=False)

    print(f"Saved {len(results_df)} search results to {output_path}")


if __name__ == "__main__":
    main()

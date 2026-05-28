import pandas as pd

from georeset_osm_web_evidence.search.coverage import (
    load_existing_search_attempts,
    load_existing_search_results,
    summarize_search_coverage,
    unsearched_polygons,
)
from georeset_osm_web_evidence.storage.local import load_geodataframe


def main() -> None:
    polygons_path = "data/processed/samples/balanced_wikipedia_100.parquet"
    search_results_path = "data/processed/search/brave_results_sample.parquet"
    attempts_path = "data/processed/search/brave_search_attempts.parquet"

    polygons_gdf = load_geodataframe(polygons_path)
    search_results_df = load_existing_search_results(search_results_path)
    attempts_df = load_existing_search_attempts(attempts_path)

    summary = summarize_search_coverage(
        polygons_gdf,
        search_results_df,
        attempted_polygons_df=attempts_df,
    )
    unsearched_df = unsearched_polygons(
        polygons_gdf,
        search_results_df,
        attempted_polygons_df=attempts_df,
    )

    print("Brave search coverage")
    print("---------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nUnsearched polygons by Wikipedia status:")
    print(unsearched_df["has_wikipedia_articles"].value_counts(dropna=False))

    print("\nUnsearched polygons by OSM type:")
    print(unsearched_df["osm_type"].value_counts(dropna=False))

    print("\nNext unsearched polygons:")
    columns = ["osm_type", "osm_id", "has_wikipedia_articles"]

    if "osm_tags" in unsearched_df.columns:
        unsearched_df = unsearched_df.copy()
        unsearched_df["polygon_name"] = unsearched_df["osm_tags"].apply(
            lambda tags: tags.get("name") if isinstance(tags, dict) else None
        )
        columns.append("polygon_name")

    print(pd.DataFrame(unsearched_df[columns].head(20)).to_string(index=False))


if __name__ == "__main__":
    main()

from georeset_osm_web_evidence.search.queries import build_search_queries
from georeset_osm_web_evidence.storage.local import load_geodataframe


def main():
    input_path = (
        "data/processed/osm_polygons_sample100_wikipedia_balanced_50_50.parquet"
    )

    gdf = load_geodataframe(input_path)

    for index, row in enumerate(gdf.itertuples()):
        osm_tags = row.osm_tags
        queries = build_search_queries(osm_tags)
        print(f"Index: {index}, Queries: {queries}")


if __name__ == "__main__":
    main()

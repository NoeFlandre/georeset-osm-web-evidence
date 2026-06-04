from georeset_osm_web_evidence.search.config import (
    BALANCED_POLYGONS_PATH,
    SEARCH_LANGUAGES,
)
from georeset_osm_web_evidence.search.queries import build_search_queries
from georeset_osm_web_evidence.storage.local import load_geodataframe


def main():
    gdf = load_geodataframe(BALANCED_POLYGONS_PATH)

    for index, row in enumerate(gdf.itertuples()):
        osm_tags = row.osm_tags
        queries = build_search_queries(osm_tags, search_languages=SEARCH_LANGUAGES)
        print(f"Index: {index}, Queries: {queries}")


if __name__ == "__main__":
    main()

import time

from georeset_osm_web_evidence.search.config import (
    BALANCED_POLYGONS_PATH,
    SEARCH_LANGUAGES,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import build_search_queries
from georeset_osm_web_evidence.storage.local import load_geodataframe


def main() -> None:
    result_count = 5
    polygon_count = 3

    gdf = load_geodataframe(BALANCED_POLYGONS_PATH)

    for row in gdf.head(polygon_count).itertuples():
        name = row.osm_tags["name"]
        query = build_search_queries(row.osm_tags, search_languages=SEARCH_LANGUAGES)[0]
        results = search_brave(query, count=result_count)

        print()
        print(f"{name}: {query}")

        for result in results:
            print(f"- {result['title']}")
            print(f"  {result['url']}")

        time.sleep(1.2)


if __name__ == "__main__":
    main()

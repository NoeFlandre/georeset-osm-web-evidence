from georeset_osm_web_evidence.osm.overpass import (
    build_polygon_query,
    fetch_overpass_json,
)
from georeset_osm_web_evidence.osm.tags import ENVIRONMENTAL_TAGS


def main() -> None:
    south = 47.7
    west = 1.5
    north = 48.1
    east = 2.2

    query = build_polygon_query(
        south=south,
        west=west,
        north=north,
        east=east,
        tags=ENVIRONMENTAL_TAGS,
    )

    data = fetch_overpass_json(query)
    elements = data.get("elements", [])
    print(f"Fetched {len(elements)} OSM elements")

    for element in elements[:2]:
        print(element["type"], element["id"], element.get("tags", {}))


if __name__ == "__main__":
    main()

from georeset_osm_web_evidence.osm.geometry import elements_to_records
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

    if not elements:
        print("No elements found")
        return

    print(f"Fetched {len(elements)} OSM elements")

    records = elements_to_records(elements)
    if not records:
        print(f"No records")
        return
    print(f"Converted {len(records)} OSM elements into records")

    first_record = records[0]
    print(f"First record: {first_record}")


if __name__ == "__main__":
    main()

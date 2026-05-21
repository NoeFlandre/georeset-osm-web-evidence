from georeset_osm_web_evidence.osm.geodataframe import (
    add_area_km2,
    add_centroid_coordinates,
    filter_by_area,
    records_to_geodataframe,
)
from georeset_osm_web_evidence.osm.geometry import elements_to_records
from georeset_osm_web_evidence.osm.overpass import (
    build_polygon_query,
    fetch_overpass_json,
)
from georeset_osm_web_evidence.osm.tags import ENVIRONMENTAL_TAGS
from georeset_osm_web_evidence.storage.local import save_geodataframe


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

    gdf = records_to_geodataframe(records)
    gdf = add_area_km2(gdf)
    gdf = filter_by_area(gdf, min_area_km2=2, max_area_km2=80)
    print(f"Kept {len(gdf)} polygons after filtering by area")

    gdf = add_centroid_coordinates(gdf)

    print(
        gdf[["osm_type", "osm_id", "area_km2", "centroid_lon", "centroid_lat"]].head()
    )

    path = "data/raw/osm_polygons_sample.parquet"
    save_geodataframe(gdf, path)
    print(f"Saved geodataframe to {path}")


if __name__ == "__main__":
    main()

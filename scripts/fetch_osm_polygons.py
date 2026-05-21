import pandas as pd

from georeset_osm_web_evidence.osm.bboxes import FRANCE_TEST_BBOXES
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
    gdfs = []
    for index, bbox in enumerate(FRANCE_TEST_BBOXES, start=1):
        south, west, north, east = bbox
        print(f"Fetching bbox {index}/{len(FRANCE_TEST_BBOXES)}:{bbox}")
        query = build_polygon_query(
            south=south,
            west=west,
            north=north,
            east=east,
            tags=ENVIRONMENTAL_TAGS,
        )

        try:
            data = fetch_overpass_json(query)
        except Exception as error:
            print(f"Skipping bbox {index} because Overpass failed : {error}")
            continue

        elements = data.get("elements", [])

        if not elements:
            print("No elements found")
            continue

        print(f"Fetched {len(elements)} OSM elements")

        records = elements_to_records(elements)
        if not records:
            print(f"No records")
            continue
        print(f"Converted {len(records)} OSM elements into records")

        gdf = records_to_geodataframe(records)

        if gdf.empty:
            print("No geodataframe")
            continue

        gdf = add_area_km2(gdf)
        gdf = filter_by_area(gdf, min_area_km2=2, max_area_km2=80)
        print(f"Kept {len(gdf)} polygons after filtering by area")

        gdf = add_centroid_coordinates(gdf)
        gdfs.append(gdf)
        print(
            gdf[
                ["osm_type", "osm_id", "area_km2", "centroid_lon", "centroid_lat"]
            ].head()
        )

    if not gdfs:
        print("No geodataframes collected")
        return

    combined_gdfs = pd.concat(gdfs, ignore_index=True)
    combined_gdfs = combined_gdfs.drop_duplicates(subset=["osm_type", "osm_id"])

    path = "data/raw/osm_polygons_sample.parquet"
    save_geodataframe(combined_gdfs, path)
    print(f"Saved {len(combined_gdfs)} candidate polygons to {path}")


if __name__ == "__main__":
    main()

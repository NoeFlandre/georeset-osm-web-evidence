from georeset_osm_web_evidence.storage.local import load_geodataframe
from georeset_osm_web_evidence.viz.map import create_polygon_map


def main() -> None:
    input_path = (
        "data/processed/osm_polygons_sample100_wikipedia_balanced_50_50.parquet"
    )
    output_path = "data/processed/osm_polygons_sample100_wikipedia_balanced_50_50.html"

    gdf = load_geodataframe(input_path)

    map_gdf = gdf[
        [
            "osm_type",
            "osm_id",
            "area_km2",
            "centroid_lon",
            "centroid_lat",
            "has_wikipedia_articles",
            "geometry",
        ]
    ].copy()

    create_polygon_map(map_gdf, output_path)


if __name__ == "__main__":
    main()

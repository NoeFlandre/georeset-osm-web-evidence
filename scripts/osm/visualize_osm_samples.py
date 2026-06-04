from georeset_osm_web_evidence.storage.local import load_geodataframe
from georeset_osm_web_evidence.viz.map import create_polygon_map


def main() -> None:
    input_path = "data/processed/samples/balanced_wikipedia_100.parquet"
    output_path = "data/processed/maps/balanced_wikipedia_100.html"

    gdf = load_geodataframe(input_path)

    map_columns = [
        "polygon_name",
        "osm_type",
        "osm_id",
        "osm_tags",
        "area_km2",
        "centroid_lon",
        "centroid_lat",
        "has_wikipedia_articles",
        "geometry",
    ]
    available_columns = [column for column in map_columns if column in gdf.columns]
    map_gdf = gdf[available_columns].copy()

    create_polygon_map(
        map_gdf,
        output_path,
        color_by="has_wikipedia_articles",
        title="Balanced Wikipedia polygon sample",
    )


if __name__ == "__main__":
    main()

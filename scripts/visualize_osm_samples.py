from georeset_osm_web_evidence.storage.local import load_geodataframe
from georeset_osm_web_evidence.viz.map import create_polygon_map


def main() -> None:
    input_path = "data/raw/osm_polygons_sample.parquet"
    output_path = "data/raw/osm_polygons_sample.html"

    gdf = load_geodataframe(input_path)
    create_polygon_map(gdf, output_path)


if __name__ == "__main__":
    main()

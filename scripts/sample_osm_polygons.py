import geopandas as gpd

from georeset_osm_web_evidence.osm.sampling import sample_polygons
from georeset_osm_web_evidence.storage.local import (
    load_geodataframe,
    save_geodataframe,
)


def main():
    input_path = "data/raw/osm_polygons_sample.parquet"
    output_path = "data/processed/osm_polygons_sample_100.parquet"
    sample_size = 100

    gdf = load_geodataframe(input_path)
    print(f"Geodataframe read at {input_path}")
    gdf = sample_polygons(gdf, sample_size=sample_size)
    print(f"Geodataframe sampled with sample size : {sample_size}")
    save_geodataframe(gdf, output_path)
    print(f"Geodataframe sampled and saved at {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path

import folium
import geopandas as gpd


def create_polygon_map(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
) -> None:
    center_lat = gdf["centroid_lat"].mean()
    center_lon = gdf["centroid_lon"].mean()

    map_ = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
    )

    folium.GeoJson(
        gdf,
        name="OSM polygon sample",
        tooltip=folium.GeoJsonTooltip(
            fields=["osm_type", "osm_id", "area_km2"],
            aliases=["OSM Type", "OSM Id", "Area km2"],
        ),
    ).add_to(map_)

    folium.LayerControl().add_to(map_)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    map_.save(output_path)
    print(f"Created a map of the polygons at {output_path}")

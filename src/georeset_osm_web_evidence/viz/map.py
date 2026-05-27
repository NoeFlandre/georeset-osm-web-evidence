from pathlib import Path

import folium
import geopandas as gpd


def style_polygon(feature: dict) -> dict:
    has_wikipedia = feature["properties"].get("has_wikipedia_articles")

    if has_wikipedia:
        return {
            "fillColor": "#f97316",
            "color": "#c2410c",
            "weight": 2,
            "fillOpacity": 0.45,
        }

    return {
        "fillColor": "#2563eb",
        "color": "#1d4ed8",
        "weight": 2,
        "fillOpacity": 0.35,
    }


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
        style_function=style_polygon,
        tooltip=folium.GeoJsonTooltip(
            fields=["osm_type", "osm_id", "area_km2", "has_wikipedia_articles"],
            aliases=["OSM Type", "OSM Id", "Area km2", "Has Wikipedia articles"],
        ),
    ).add_to(map_)

    folium.LayerControl().add_to(map_)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    map_.save(output_path)
    print(f"Created a map of the polygons at {output_path}")

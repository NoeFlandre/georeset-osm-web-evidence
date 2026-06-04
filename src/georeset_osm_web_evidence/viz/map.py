import json
from pathlib import Path
from typing import Any

from branca.element import Element
import folium
import geopandas as gpd
import pandas as pd
from pandas.api.types import is_scalar


DEFAULT_TOOLTIP_COLUMNS = [
    "polygon_name",
    "osm_type",
    "osm_id",
    "area_km2",
    "has_wikipedia_articles",
    "area_size_bin",
    "country",
    "local_language",
]

TOOLTIP_ALIASES = {
    "polygon_name": "Polygon name",
    "osm_type": "OSM type",
    "osm_id": "OSM id",
    "area_km2": "Area km2",
    "has_wikipedia_articles": "Has Wikipedia articles",
    "area_size_bin": "Area size bin",
    "country": "Country",
    "local_language": "Local language",
}

COLOR_PALETTE = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#ca8a04",
    "#db2777",
]

OSM_NAME_KEYS = ["name", "name:fr", "name:en", "alt_name", "short_name"]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if not is_scalar(value):
        return False
    return bool(pd.isna(value))


def _make_json_ready(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "tolist"):
        return _make_json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _make_json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_make_json_ready(item) for item in value]
    return str(value)


def _extract_polygon_name(osm_tags: Any) -> str | None:
    if not isinstance(osm_tags, dict):
        return None

    for key in OSM_NAME_KEYS:
        value = osm_tags.get(key)
        if not _is_missing(value):
            return str(value)

    return None


def _serialize_property(value: Any) -> Any:
    json_ready = _make_json_ready(value)

    if isinstance(json_ready, dict | list):
        return json.dumps(json_ready, ensure_ascii=False, sort_keys=True)

    return json_ready


def prepare_map_geodataframe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        raise ValueError("Cannot create a map from an empty GeoDataFrame")

    map_gdf = gdf.copy()

    if map_gdf.crs is None:
        map_gdf = map_gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        map_gdf = map_gdf.to_crs("EPSG:4326")

    if "osm_tags" in map_gdf.columns:
        derived_names = map_gdf["osm_tags"].map(_extract_polygon_name)
        if "polygon_name" not in map_gdf.columns:
            map_gdf["polygon_name"] = derived_names
        else:
            map_gdf["polygon_name"] = [
                derived_name if _is_missing(existing_name) else existing_name
                for existing_name, derived_name in zip(
                    map_gdf["polygon_name"], derived_names, strict=True
                )
            ]

    geometry_column = map_gdf.geometry.name
    for column in map_gdf.columns:
        if column == geometry_column:
            continue
        map_gdf[column] = map_gdf[column].map(_serialize_property)

    return map_gdf


def _display_column_name(column: str) -> str:
    return TOOLTIP_ALIASES.get(column, column.replace("_", " ").capitalize())


def _display_value(value: Any) -> str:
    if _is_missing(value):
        return "missing"
    return str(value)


def _build_color_map(gdf: gpd.GeoDataFrame, color_by: str | None) -> dict[str, str]:
    if color_by is None or color_by not in gdf.columns:
        return {}

    values = [_display_value(value) for value in gdf[color_by].dropna().unique()]

    if set(values).issubset({"False", "True"}):
        return {
            "False": "#2563eb",
            "True": "#f97316",
        }

    return {
        value: COLOR_PALETTE[index % len(COLOR_PALETTE)]
        for index, value in enumerate(sorted(values))
    }


def _style_for_color(color: str) -> dict:
    return {
        "fillColor": color,
        "color": color,
        "weight": 2,
        "fillOpacity": 0.4,
    }


def _make_style_function(color_by: str | None, color_map: dict[str, str]):
    def style_function(feature: dict) -> dict:
        if color_by is None:
            return _style_for_color("#2563eb")

        value = feature["properties"].get(color_by)
        color = color_map.get(_display_value(value), "#6b7280")
        return _style_for_color(color)

    return style_function


def _get_tooltip_fields(
    gdf: gpd.GeoDataFrame, tooltip_columns: list[str] | None
) -> list[str]:
    columns = tooltip_columns or DEFAULT_TOOLTIP_COLUMNS
    geometry_column = gdf.geometry.name
    return [
        column
        for column in columns
        if column in gdf.columns and column != geometry_column
    ]


def _get_tooltip(gdf: gpd.GeoDataFrame, fields: list[str]):
    if not fields:
        return None

    return folium.GeoJsonTooltip(
        fields=fields,
        aliases=[_display_column_name(column) for column in fields],
    )


def _select_geojson_columns(
    gdf: gpd.GeoDataFrame,
    color_by: str | None,
    tooltip_fields: list[str],
) -> gpd.GeoDataFrame:
    geometry_column = gdf.geometry.name
    columns = list(dict.fromkeys([*tooltip_fields, color_by, geometry_column]))
    columns = [
        column for column in columns if column is not None and column in gdf.columns
    ]
    return gdf[columns].copy()


def _fit_map_to_bounds(map_: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    map_.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])


def _add_title(map_: folium.Map, title: str | None) -> None:
    if title is None:
        return

    title_html = f"""
    <div style="
        position: fixed;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.18);
    ">{title}</div>
    """
    map_.get_root().html.add_child(Element(title_html))


def _add_legend(map_: folium.Map, color_by: str | None, color_map: dict[str, str]) -> None:
    if color_by is None or not color_map:
        return

    rows = "\n".join(
        f"""
        <div style="display: flex; align-items: center; gap: 6px; margin-top: 4px;">
            <span style="width: 11px; height: 11px; background: {color}; display: inline-block;"></span>
            <span>{value}</span>
        </div>
        """
        for value, color in color_map.items()
    )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 10px 12px;
        font-size: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.18);
    ">
        <div style="font-weight: 600;">{_display_column_name(color_by)}</div>
        {rows}
    </div>
    """
    map_.get_root().html.add_child(Element(legend_html))


def style_polygon(feature: dict) -> dict:
    has_wikipedia = feature["properties"].get("has_wikipedia_articles")

    if has_wikipedia is True or has_wikipedia == "True":
        return _style_for_color("#f97316")

    return _style_for_color("#2563eb")


def create_polygon_map(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
    color_by: str | None = None,
    tooltip_columns: list[str] | None = None,
    title: str | None = None,
) -> None:
    map_gdf = prepare_map_geodataframe(gdf)
    resolved_color_by = color_by

    if resolved_color_by is None and "has_wikipedia_articles" in map_gdf.columns:
        resolved_color_by = "has_wikipedia_articles"

    tooltip_fields = _get_tooltip_fields(map_gdf, tooltip_columns)
    map_gdf = _select_geojson_columns(map_gdf, resolved_color_by, tooltip_fields)

    min_lon, min_lat, max_lon, max_lat = map_gdf.total_bounds
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    color_map = _build_color_map(map_gdf, resolved_color_by)

    map_ = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
    )

    folium.GeoJson(
        map_gdf,
        name="OSM polygon sample",
        style_function=_make_style_function(resolved_color_by, color_map),
        tooltip=_get_tooltip(map_gdf, tooltip_fields),
    ).add_to(map_)

    _fit_map_to_bounds(map_, map_gdf)
    _add_title(map_, title)
    _add_legend(map_, resolved_color_by, color_map)
    folium.LayerControl().add_to(map_)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    map_.save(output_path)
    print(f"Created a map of the polygons at {output_path}")

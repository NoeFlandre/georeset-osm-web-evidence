import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from georeset_osm_web_evidence.osm.tags import ENVIRONMENTAL_TAGS

TAG_COLUMNS = [
    "name",
    "landuse",
    "natural",
    "leisure",
    "boundary",
    "other_tags",
]


def build_environmental_where_clause(
    tags: list[tuple[str, str]] = ENVIRONMENTAL_TAGS,
) -> str:
    values_by_key: dict[str, list[str]] = {}
    for key, value in tags:
        values_by_key.setdefault(key, []).append(value)

    clauses = []
    for key, values in values_by_key.items():
        quoted_values = ", ".join(f"'{value}'" for value in values)
        if len(values) == 1:
            clauses.append(f"{key} = {quoted_values}")
        else:
            clauses.append(f"{key} IN ({quoted_values})")

    return f"name IS NOT NULL AND ({' OR '.join(clauses)})"


def read_pbf_multipolygons(
    pbf_path: str | Path,
) -> gpd.GeoDataFrame:
    return gpd.read_file(
        pbf_path,
        layer="multipolygons",
        where=build_environmental_where_clause(),
    )


def _is_present(value: Any) -> bool:
    return pd.notna(value) and value != ""


def _coerce_osm_id(value: Any) -> int | str:
    value_as_text = str(value)

    return int(value_as_text) if value_as_text.isdigit() else value_as_text


def _row_to_osm_tags(row: pd.Series) -> dict:
    return {
        column: row[column]
        for column in TAG_COLUMNS
        if column in row.index and _is_present(row[column])
    }


def multipolygons_to_candidate_gdf(
    multipolygons_gdf: gpd.GeoDataFrame,
    extract_config: dict,
) -> gpd.GeoDataFrame:
    records = []

    for _, row in multipolygons_gdf.iterrows():
        if _is_present(row.get("osm_id")):
            osm_type = "relation"
            osm_id = _coerce_osm_id(row["osm_id"])
        elif _is_present(row.get("osm_way_id")):
            osm_type = "way"
            osm_id = _coerce_osm_id(row["osm_way_id"])
        else:
            continue

        records.append(
            {
                "osm_type": osm_type,
                "osm_id": osm_id,
                "osm_tags": _row_to_osm_tags(row),
                "source_extract_id": extract_config["extract_id"],
                "bbox_label": extract_config["extract_label"],
                "country": extract_config["country"],
                "world_region": extract_config["world_region"],
                "local_language": extract_config["local_language"],
                "geometry": row.geometry,
            }
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=multipolygons_gdf.crs,
    )


def add_extract_spatial_cells(
    gdf: gpd.GeoDataFrame,
    cell_size_degrees: float = 1.0,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    representative_points = gdf.geometry.apply(lambda geometry: geometry.representative_point())
    lon_cells = representative_points.apply(lambda point: math.floor(point.x / cell_size_degrees))
    lat_cells = representative_points.apply(lambda point: math.floor(point.y / cell_size_degrees))

    gdf["bbox_id"] = [
        f"extract:{source_extract_id}:cell:lat{lat_cell}_lon{lon_cell}"
        for source_extract_id, lat_cell, lon_cell in zip(
            gdf["source_extract_id"],
            lat_cells,
            lon_cells,
        )
    ]

    return gdf

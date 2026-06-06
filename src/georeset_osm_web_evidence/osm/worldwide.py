import pandas as pd
import geopandas as gpd

from georeset_osm_web_evidence.osm.worldwide_balancing import compute_group_targets
from georeset_osm_web_evidence.osm.worldwide_bboxes import (
    BASE_WORLDWIDE_TRAINING_BBOX_COUNT,
    WORLDWIDE_PILOT_BBOXES,
    WORLDWIDE_TRAINING_BBOXES,
    bbox_config,
    generate_bbox_expansions,
)
from georeset_osm_web_evidence.osm.spatial_distance import (
    add_point_to_distance_grid,
    distance_cell_size_degrees,
    geodesic_distance_km,
    is_far_enough_from_distance_grid,
)
from georeset_osm_web_evidence.osm.tags import ENVIRONMENTAL_TAGS


def compute_sample_size(
    target_sentences: int,
    planned_sentences_per_polygon: int,
) -> int:
    if planned_sentences_per_polygon <= 0:
        raise ValueError("planned_sentences_per_polygon must be positive")

    return -(-target_sentences // planned_sentences_per_polygon)


def _has_usable_name(osm_tags: dict) -> bool:
    name = osm_tags.get("name")

    return isinstance(name, str) and name.strip() != ""


def _has_environmental_tag(osm_tags: dict) -> bool:
    return any(osm_tags.get(key) == value for key, value in ENVIRONMENTAL_TAGS)


def filter_named_environmental_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "osm_tags" not in gdf.columns:
        return gdf.head(0).copy()

    keep_mask = gdf["osm_tags"].apply(
        lambda osm_tags: isinstance(osm_tags, dict)
        and _has_usable_name(osm_tags)
        and _has_environmental_tag(osm_tags)
    )

    return gdf[keep_mask].copy().reset_index(drop=True)


def add_bbox_metadata(
    gdf: gpd.GeoDataFrame,
    bbox_config: dict,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    for column in [
        "bbox_id",
        "bbox_label",
        "country",
        "world_region",
        "local_language",
    ]:
        gdf[column] = bbox_config[column]

    return gdf


def add_area_size_bin(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["area_size_bin"] = pd.cut(
        gdf["area_km2"],
        bins=[0, 0.1, 1, 10, float("inf")],
        labels=["tiny", "small", "medium", "large"],
        right=False,
        include_lowest=True,
    ).astype(str)

    return gdf


def _balanced_downsample(
    gdf: gpd.GeoDataFrame,
    sample_size: int,
    group_columns: list[str],
    random_state: int | None,
) -> gpd.GeoDataFrame:
    if not group_columns:
        return gdf.sample(n=sample_size, random_state=random_state)

    groups = list(gdf.groupby(group_columns, sort=True, dropna=False))
    base_quota = sample_size // len(groups)
    remainder = sample_size % len(groups)

    sampled_parts = []
    sampled_indices = []
    for group_index, (_, group) in enumerate(groups):
        quota = base_quota + int(group_index < remainder)
        if quota == 0:
            continue

        seed = None if random_state is None else random_state + group_index
        n = min(quota, len(group))
        if "_spatial_priority" in group.columns:
            part = group.sort_values("_spatial_priority").head(n)
        else:
            part = group.sample(n=n, random_state=seed)
        sampled_parts.append(part)
        sampled_indices.extend(part.index.to_list())

    sample = pd.concat(sampled_parts) if sampled_parts else gdf.head(0)

    if len(sample) < sample_size:
        remaining = gdf.drop(index=sampled_indices)
        fill_n = min(sample_size - len(sample), len(remaining))
        if fill_n > 0:
            fill = remaining.sample(n=fill_n, random_state=random_state)
            sample = pd.concat([sample, fill])

    return gpd.GeoDataFrame(
        sample,
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )


def _cap_group_size(
    gdf: gpd.GeoDataFrame,
    group_column: str,
    max_per_group: int | None,
    random_state: int | None,
) -> gpd.GeoDataFrame:
    if max_per_group is None or group_column not in gdf.columns:
        return gdf

    if max_per_group <= 0:
        return gdf.head(0).copy()

    capped_parts = []
    for group_index, (_, group) in enumerate(gdf.groupby(group_column, sort=True)):
        seed = None if random_state is None else random_state + group_index
        n = min(max_per_group, len(group))
        capped_parts.append(group.sample(n=n, random_state=seed))

    capped = pd.concat(capped_parts) if capped_parts else gdf.head(0)

    return gpd.GeoDataFrame(
        capped,
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )


def _group_columns_for_balancing(gdf: gpd.GeoDataFrame) -> list[str]:
    return [
        column
        for column in ["world_region", "area_size_bin"]
        if column in gdf.columns
    ]


def _group_key(row: pd.Series, group_columns: list[str]) -> tuple:
    if not group_columns:
        return ("__all__",)

    return tuple(row[column] for column in group_columns)


def _select_balanced_sparse_rows(
    gdf: gpd.GeoDataFrame,
    sample_size: int,
    max_per_bbox: int | None,
    max_per_country: int | None,
    min_centroid_distance_km: float,
    min_global_centroid_distance_km: float,
    random_state: int | None,
) -> gpd.GeoDataFrame:
    gdf = _add_representative_coordinates(gdf)
    group_columns = _group_columns_for_balancing(gdf)
    group_targets = compute_group_targets(gdf, sample_size, group_columns)

    grouped_rows = {}
    for group_index, (group_key, group) in enumerate(
        gdf.groupby(group_columns, sort=True, dropna=False)
        if group_columns
        else [(("__all__",), gdf)]
    ):
        normalized_key = group_key if isinstance(group_key, tuple) else (group_key,)
        seed = None if random_state is None else random_state + group_index
        grouped_rows[normalized_key] = list(group.sample(frac=1, random_state=seed).iterrows())

    grouped_positions = dict.fromkeys(grouped_rows, 0)
    selected_rows = []
    selected_indices = set()
    selected_group_counts = dict.fromkeys(group_targets, 0)
    selected_bbox_counts = {}
    selected_country_counts = {}
    selected_bbox_grids = {}
    selected_global_grid = {}
    local_cell_size_degrees = distance_cell_size_degrees(min_centroid_distance_km)
    global_cell_size_degrees = distance_cell_size_degrees(
        min_global_centroid_distance_km
    )

    def can_select(
        index,
        row,
        enforce_local_distance: bool,
        enforce_global_distance: bool,
    ) -> bool:
        if index in selected_indices:
            return False

        bbox_id = row.get("bbox_id")
        if (
            max_per_bbox is not None
            and bbox_id is not None
            and selected_bbox_counts.get(bbox_id, 0) >= max_per_bbox
        ):
            return False

        country = row.get("country")
        if (
            max_per_country is not None
            and country is not None
            and selected_country_counts.get(country, 0) >= max_per_country
        ):
            return False

        lon = row["centroid_lon"]
        lat = row["centroid_lat"]
        if (
            enforce_local_distance
            and min_centroid_distance_km > 0
            and bbox_id is not None
            and not is_far_enough_from_distance_grid(
                lon,
                lat,
                selected_bbox_grids.get(bbox_id, {}),
                local_cell_size_degrees,
                min_centroid_distance_km,
            )
        ):
            return False

        if (
            enforce_global_distance
            and min_global_centroid_distance_km > 0
            and not is_far_enough_from_distance_grid(
                lon,
                lat,
                selected_global_grid,
                global_cell_size_degrees,
                min_global_centroid_distance_km,
            )
        ):
            return False

        return True

    def select(index, row) -> None:
        selected_rows.append(row)
        selected_indices.add(index)
        group_key = _group_key(row, group_columns)
        selected_group_counts[group_key] = selected_group_counts.get(group_key, 0) + 1

        bbox_id = row.get("bbox_id")
        if bbox_id is not None:
            selected_bbox_counts[bbox_id] = selected_bbox_counts.get(bbox_id, 0) + 1
            selected_bbox_grids.setdefault(bbox_id, {})
            add_point_to_distance_grid(
                selected_bbox_grids[bbox_id],
                row["centroid_lon"],
                row["centroid_lat"],
                local_cell_size_degrees,
            )

        country = row.get("country")
        if country is not None:
            selected_country_counts[country] = selected_country_counts.get(country, 0) + 1

        add_point_to_distance_grid(
            selected_global_grid,
            row["centroid_lon"],
            row["centroid_lat"],
            global_cell_size_degrees,
        )

    def pick_next_for_group(
        group_key: tuple,
        enforce_local_distance: bool,
        enforce_global_distance: bool,
    ) -> bool:
        rows = grouped_rows.get(group_key, [])
        position = grouped_positions.get(group_key, 0)
        while position < len(rows):
            index, row = rows[position]
            grouped_positions[group_key] = position + 1
            position += 1
            if can_select(index, row, enforce_local_distance, enforce_global_distance):
                select(index, row)
                return True

        return False

    def pick_any_for_group(
        group_key: tuple,
        enforce_local_distance: bool,
        enforce_global_distance: bool,
    ) -> bool:
        for index, row in grouped_rows.get(group_key, []):
            if can_select(index, row, enforce_local_distance, enforce_global_distance):
                select(index, row)
                return True

        return False

    def sorted_under_target_group_keys() -> list[tuple]:
        eligible_group_keys = [
            group_key
            for group_key, target in group_targets.items()
            if selected_group_counts.get(group_key, 0) < target
        ]
        eligible_group_keys.sort(
            key=lambda group_key: (
                selected_group_counts.get(group_key, 0) / max(group_targets[group_key], 1),
                group_key,
            )
        )

        return eligible_group_keys

    while len(selected_rows) < min(sample_size, len(gdf)):
        eligible_group_keys = sorted_under_target_group_keys()
        if not eligible_group_keys:
            break

        made_progress = False
        for group_key in eligible_group_keys:
            if pick_next_for_group(
                group_key,
                enforce_local_distance=True,
                enforce_global_distance=True,
            ):
                made_progress = True
                break

        if not made_progress:
            break

    while len(selected_rows) < min(sample_size, len(gdf)):
        eligible_group_keys = sorted_under_target_group_keys()
        if not eligible_group_keys:
            break

        made_progress = False
        for group_key in eligible_group_keys:
            if pick_any_for_group(
                group_key,
                enforce_local_distance=False,
                enforce_global_distance=True,
            ):
                made_progress = True
                break

        if not made_progress:
            break

    if len(selected_rows) < min(sample_size, len(gdf)):
        shuffled = gdf.sample(frac=1, random_state=random_state)
        for index, row in shuffled.iterrows():
            if len(selected_rows) >= min(sample_size, len(gdf)):
                break

            if can_select(
                index,
                row,
                enforce_local_distance=False,
                enforce_global_distance=True,
            ):
                select(index, row)

    selected = (
        gpd.GeoDataFrame(selected_rows, geometry=gdf.geometry.name, crs=gdf.crs)
        if selected_rows
        else gdf.head(0).copy()
    )

    return selected


def _add_representative_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if {"centroid_lon", "centroid_lat"}.issubset(gdf.columns):
        return gdf

    gdf = gdf.copy()
    representative_points = gdf.geometry.apply(lambda geometry: geometry.representative_point())
    gdf["centroid_lon"] = representative_points.apply(lambda point: point.x)
    gdf["centroid_lat"] = representative_points.apply(lambda point: point.y)

    return gdf


def _spatially_thin(
    gdf: gpd.GeoDataFrame,
    target_size: int,
    min_centroid_distance_km: float,
    random_state: int | None,
    fill_shortfall: bool = True,
) -> gpd.GeoDataFrame:
    if min_centroid_distance_km <= 0:
        n = min(target_size, len(gdf))
        sample = gdf.sample(n=n, random_state=random_state)
        sample["_spatial_priority"] = 0
        return sample

    gdf = _add_representative_coordinates(gdf)
    shuffled = gdf.sample(frac=1, random_state=random_state)

    selected_rows = []
    selected_indices = []
    selected_points = []
    for index, row in shuffled.iterrows():
        lon = row["centroid_lon"]
        lat = row["centroid_lat"]
        is_far_enough = all(
            geodesic_distance_km(lon, lat, selected_lon, selected_lat)
            >= min_centroid_distance_km
            for selected_lon, selected_lat in selected_points
        )

        if not is_far_enough:
            continue

        selected_rows.append(row)
        selected_indices.append(index)
        selected_points.append((lon, lat))

        if len(selected_rows) >= target_size:
            break

    selected = (
        gpd.GeoDataFrame(selected_rows, geometry=gdf.geometry.name, crs=gdf.crs)
        if selected_rows
        else gdf.head(0)
    )
    selected["_spatial_priority"] = 0

    if fill_shortfall and len(selected) < min(target_size, len(gdf)):
        remaining = shuffled.drop(index=selected_indices)
        fill_n = min(target_size - len(selected), len(remaining))
        if fill_n > 0:
            fill = remaining.sample(n=fill_n, random_state=random_state)
            fill["_spatial_priority"] = 1
            selected = pd.concat([selected, fill])

    return gpd.GeoDataFrame(
        selected,
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )


def sample_worldwide_polygons(
    gdf: gpd.GeoDataFrame,
    sample_size: int = 100,
    max_per_bbox: int = 8,
    max_per_country: int | None = None,
    min_centroid_distance_km: float = 0,
    min_global_centroid_distance_km: float = 0,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    sample = _select_balanced_sparse_rows(
        gdf,
        sample_size=sample_size,
        max_per_bbox=max_per_bbox,
        max_per_country=max_per_country,
        min_centroid_distance_km=min_centroid_distance_km,
        min_global_centroid_distance_km=min_global_centroid_distance_km,
        random_state=random_state,
    )

    if "_spatial_priority" in sample.columns:
        sample = sample.drop(columns=["_spatial_priority"])

    return sample.reset_index(drop=True)

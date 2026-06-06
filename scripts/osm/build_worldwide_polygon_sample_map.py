import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from georeset_osm_web_evidence.osm.geodataframe import (
    add_geodesic_area_km2,
    filter_by_area,
    records_to_geodataframe,
)
from georeset_osm_web_evidence.osm.geometry import (
    elements_to_records,
    filter_records_with_name,
)
from georeset_osm_web_evidence.osm.overpass import (
    build_polygon_query,
    fetch_overpass_json,
)
from georeset_osm_web_evidence.osm.tags import ENVIRONMENTAL_TAGS
from georeset_osm_web_evidence.osm.worldwide import (
    BASE_WORLDWIDE_TRAINING_BBOX_COUNT,
    WORLDWIDE_TRAINING_BBOXES,
    add_area_size_bin,
    add_bbox_metadata,
    compute_sample_size,
    filter_named_environmental_polygons,
    sample_worldwide_polygons,
)
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.viz.map import create_polygon_map


RAW_OUTPUT_PATH = "data/raw/osm/worldwide_named_polygon_candidates.parquet"
ATTEMPTED_BBOX_OUTPUT_PATH = "data/raw/osm/worldwide_attempted_bbox_ids.txt"
SAMPLE_OUTPUT_PATH = "data/processed/samples/worldwide_training_polygon_sample.parquet"
MAP_OUTPUT_PATH = "data/processed/maps/worldwide_training_polygon_sample.html"

TARGET_TRAINING_SENTENCES = 50_000
PLANNED_SENTENCES_PER_POLYGON = 10
MIN_AREA_KM2 = 0.02
MAX_AREA_KM2 = 100
SAMPLE_SIZE = compute_sample_size(
    TARGET_TRAINING_SENTENCES,
    PLANNED_SENTENCES_PER_POLYGON,
)
MAX_PER_BBOX = 1
MAX_PER_COUNTRY = 100
REQUEST_PAUSE_SECONDS = float(
    os.environ.get("WORLDWIDE_OSM_REQUEST_PAUSE_SECONDS", "1.5")
)
SPARSITY_DISTANCE_KM_STEPS = [100, 80, 60, 40, 25]
MAX_MISSING_BBOX_FETCHES_PER_RUN = int(
    os.environ.get("WORLDWIDE_OSM_MAX_BBOX_FETCHES_PER_RUN", "250")
)

TOOLTIP_COLUMNS = [
    "polygon_name",
    "country",
    "world_region",
    "local_language",
    "bbox_label",
    "osm_type",
    "osm_id",
    "area_km2",
    "area_size_bin",
]


def fetch_bbox_candidates(bbox_config: dict) -> gpd.GeoDataFrame | None:
    south, west, north, east = bbox_config["bbox"]
    query = build_polygon_query(
        south=south,
        west=west,
        north=north,
        east=east,
        tags=ENVIRONMENTAL_TAGS,
        require_name=True,
    )

    data = fetch_overpass_json(query, max_retries=2, retry_delay_seconds=8)
    elements = data.get("elements", [])

    if not elements:
        return None

    print(f"Fetched {len(elements)} OSM elements")
    records = elements_to_records(elements)
    records = filter_records_with_name(records)

    if not records:
        return None

    print(f"Kept {len(records)} named OSM records")
    gdf = records_to_geodataframe(records)
    gdf = add_geodesic_area_km2(gdf)
    gdf = filter_by_area(gdf, min_area_km2=MIN_AREA_KM2, max_area_km2=MAX_AREA_KM2)

    if gdf.empty:
        return None

    gdf = add_area_size_bin(gdf)
    gdf = add_bbox_metadata(gdf, bbox_config)
    gdf = filter_named_environmental_polygons(gdf)

    return gdf


def combine_candidate_gdfs(gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    prepared_gdfs = [prepare_candidate_pool(gdf) for gdf in gdfs if not gdf.empty]
    prepared_gdfs = [gdf for gdf in prepared_gdfs if not gdf.empty]

    if not prepared_gdfs:
        return gpd.GeoDataFrame(geometry="geometry", crs="EPSG:4326")

    combined = gpd.GeoDataFrame(
        pd.concat(prepared_gdfs, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    return combined.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(
        drop=True
    )


def save_candidate_gdfs(gdfs: list[gpd.GeoDataFrame], path: str) -> gpd.GeoDataFrame:
    candidates_gdf = combine_candidate_gdfs(gdfs)
    save_geodataframe(candidates_gdf, path)

    return candidates_gdf


def prepare_candidate_pool(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if gdf.empty:
        return gdf

    if "area_km2" not in gdf.columns:
        gdf = add_geodesic_area_km2(gdf)

    gdf = filter_by_area(
        gdf,
        min_area_km2=MIN_AREA_KM2,
        max_area_km2=MAX_AREA_KM2,
    )

    if gdf.empty:
        return gdf

    gdf = add_area_size_bin(gdf)
    gdf = filter_named_environmental_polygons(gdf)

    return gdf.reset_index(drop=True)


def load_existing_candidates(path: str) -> gpd.GeoDataFrame | None:
    if not Path(path).exists():
        return None

    gdf = prepare_candidate_pool(load_geodataframe(path))
    if gdf.empty:
        return None

    return gdf


def load_attempted_bbox_ids(path: str) -> set[str]:
    attempted_path = Path(path)
    if not attempted_path.exists():
        return set()

    return {
        line.strip()
        for line in attempted_path.read_text().splitlines()
        if line.strip()
    }


def save_attempted_bbox_ids(bbox_ids: set[str], path: str) -> None:
    attempted_path = Path(path)
    attempted_path.parent.mkdir(parents=True, exist_ok=True)
    attempted_path.write_text("\n".join(sorted(bbox_ids)) + "\n")


def sample_with_relaxed_sparsity(candidates_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    latest_sample = candidates_gdf.head(0).copy()

    for min_distance_km in SPARSITY_DISTANCE_KM_STEPS:
        sample_gdf = sample_worldwide_polygons(
            candidates_gdf,
            sample_size=SAMPLE_SIZE,
            max_per_bbox=MAX_PER_BBOX,
            max_per_country=MAX_PER_COUNTRY,
            min_centroid_distance_km=min_distance_km,
            min_global_centroid_distance_km=min_distance_km,
            random_state=42,
        )
        latest_sample = sample_gdf
        print(
            f"Sparsity pass with {min_distance_km} km minimum centroid distance "
            f"kept {len(sample_gdf)} polygons"
        )

        if len(sample_gdf) >= SAMPLE_SIZE:
            return sample_gdf

    return latest_sample


def main() -> None:
    candidate_gdfs = []
    failed_bbox_ids = []
    existing_gdf = load_existing_candidates(RAW_OUTPUT_PATH)
    attempted_bbox_ids = load_attempted_bbox_ids(ATTEMPTED_BBOX_OUTPUT_PATH)
    existing_bbox_ids = set()

    if existing_gdf is not None:
        candidate_gdfs.append(existing_gdf)
        existing_bbox_ids = set(existing_gdf["bbox_id"].dropna().unique())
        print(
            f"Loaded {len(existing_gdf)} existing candidates from "
            f"{RAW_OUTPUT_PATH}"
        )

    bboxes_to_consider = WORLDWIDE_TRAINING_BBOXES
    if existing_gdf is not None:
        bboxes_to_consider = WORLDWIDE_TRAINING_BBOXES[
            BASE_WORLDWIDE_TRAINING_BBOX_COUNT:
        ]

    missing_bbox_configs = [
        bbox_config
        for bbox_config in bboxes_to_consider
        if bbox_config["bbox_id"] not in existing_bbox_ids
        and bbox_config["bbox_id"] not in attempted_bbox_ids
    ]

    if missing_bbox_configs:
        if len(missing_bbox_configs) > MAX_MISSING_BBOX_FETCHES_PER_RUN:
            print(
                f"Fetching the first {MAX_MISSING_BBOX_FETCHES_PER_RUN} of "
                f"{len(missing_bbox_configs)} missing new bboxes in this run"
            )
            missing_bbox_configs = missing_bbox_configs[
                :MAX_MISSING_BBOX_FETCHES_PER_RUN
            ]

        for index, bbox_config in enumerate(missing_bbox_configs, start=1):
            print(
                f"Fetching missing bbox {index}/{len(missing_bbox_configs)}: "
                f"{bbox_config['bbox_id']} ({bbox_config['country']})"
            )

            try:
                gdf = fetch_bbox_candidates(bbox_config)
            except requests.RequestException as error:
                print(f"Failed bbox {bbox_config['bbox_id']}: {error}")
                failed_bbox_ids.append(bbox_config["bbox_id"])
                attempted_bbox_ids.add(bbox_config["bbox_id"])
                save_attempted_bbox_ids(
                    attempted_bbox_ids,
                    ATTEMPTED_BBOX_OUTPUT_PATH,
                )
                continue

            if gdf is None or gdf.empty:
                print("No usable polygons after filtering")
                attempted_bbox_ids.add(bbox_config["bbox_id"])
                save_attempted_bbox_ids(
                    attempted_bbox_ids,
                    ATTEMPTED_BBOX_OUTPUT_PATH,
                )
                time.sleep(REQUEST_PAUSE_SECONDS)
                continue

            print(f"Kept {len(gdf)} polygons after filtering by area")
            print(
                gdf[
                    [
                        "bbox_id",
                        "country",
                        "world_region",
                        "osm_type",
                        "osm_id",
                        "area_km2",
                        "area_size_bin",
                    ]
                ].head()
            )
            candidate_gdfs.append(gdf)
            attempted_bbox_ids.add(bbox_config["bbox_id"])
            save_attempted_bbox_ids(
                attempted_bbox_ids,
                ATTEMPTED_BBOX_OUTPUT_PATH,
            )
            save_candidate_gdfs(candidate_gdfs, RAW_OUTPUT_PATH)
            time.sleep(REQUEST_PAUSE_SECONDS)
    else:
        print(
            "All configured bboxes already have cached candidates; "
            "skipping Overpass fetches and resampling locally."
        )

    if not candidate_gdfs:
        raise RuntimeError("No worldwide OSM polygon candidates were collected")

    candidates_gdf = save_candidate_gdfs(candidate_gdfs, RAW_OUTPUT_PATH)
    print(f"Saved {len(candidates_gdf)} candidates to {RAW_OUTPUT_PATH}")

    sample_gdf = sample_with_relaxed_sparsity(candidates_gdf)
    save_geodataframe(sample_gdf, SAMPLE_OUTPUT_PATH)
    print(f"Saved {len(sample_gdf)} sampled polygons to {SAMPLE_OUTPUT_PATH}")
    if len(sample_gdf) < SAMPLE_SIZE:
        print(
            f"Sample is {SAMPLE_SIZE - len(sample_gdf)} polygons below the "
            f"{SAMPLE_SIZE} polygon target; collect more worldwide candidates "
            "before large-scale sentence extraction."
        )
    print(
        f"Planning estimate: {len(sample_gdf)} polygons x "
        f"{PLANNED_SENTENCES_PER_POLYGON} sentences/polygon ~= "
        f"{len(sample_gdf) * PLANNED_SENTENCES_PER_POLYGON} candidate sentences"
    )

    create_polygon_map(
        sample_gdf,
        MAP_OUTPUT_PATH,
        color_by="world_region",
        tooltip_columns=TOOLTIP_COLUMNS,
        title="Worldwide OSM polygon training sample",
    )

    print("Sample by world region:")
    print(sample_gdf["world_region"].value_counts())
    print("Sample by area size:")
    print(sample_gdf["area_size_bin"].value_counts())

    if failed_bbox_ids:
        print(f"Failed bboxes after retries: {failed_bbox_ids}")


if __name__ == "__main__":
    main()

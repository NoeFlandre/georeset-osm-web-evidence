import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from georeset_osm_web_evidence.osm.extracts import (
    add_extract_spatial_cells,
    multipolygons_to_candidate_gdf,
    read_pbf_multipolygons,
)
from georeset_osm_web_evidence.osm.geodataframe import (
    add_geodesic_area_km2,
    filter_by_area,
)
from georeset_osm_web_evidence.osm.worldwide import (
    add_area_size_bin,
    compute_sample_size,
    filter_named_environmental_polygons,
    sample_worldwide_polygons,
)
from georeset_osm_web_evidence.osm.worldwide_planning import (
    compute_region_sample_deficits,
    is_better_worldwide_sample,
    rank_extract_configs_by_region_deficit,
    region_count_spread,
)
from georeset_osm_web_evidence.osm.worldwide_extract_configs import (
    DEFAULT_LANGUAGE_BY_REGION,
    EXTRACT_CONFIGS,
    REGION_BY_GEOFABRIK_ROOT,
    SKIP_DISCOVERED_EXTRACT_IDS,
    configured_world_regions,
)
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.viz.map import create_polygon_map


GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
EXTRACT_DIR = Path("data/external/osm_extracts")
INDEX_PATH = EXTRACT_DIR / "geofabrik-index-v1.json"
OVERPASS_CANDIDATES_PATH = "data/raw/osm/worldwide_named_polygon_candidates.parquet"
EXTRACT_CANDIDATES_PATH = "data/raw/osm/worldwide_extract_named_polygon_candidates.parquet"
COMBINED_CANDIDATES_PATH = "data/raw/osm/worldwide_combined_named_polygon_candidates.parquet"
ATTEMPTED_EXTRACT_IDS_PATH = "data/raw/osm/worldwide_attempted_extract_ids.txt"
SAMPLE_OUTPUT_PATH = "data/processed/samples/worldwide_training_polygon_sample.parquet"
MAP_OUTPUT_PATH = "data/processed/maps/worldwide_training_polygon_sample.html"

TARGET_TRAINING_SENTENCES = 50_000
PLANNED_SENTENCES_PER_POLYGON = 10
SAMPLE_SIZE = compute_sample_size(
    TARGET_TRAINING_SENTENCES,
    PLANNED_SENTENCES_PER_POLYGON,
)
MIN_AREA_KM2 = 0.02
MAX_AREA_KM2 = 100
MAX_PER_BBOX = 1
MAX_PER_COUNTRY = 150
SPARSITY_DISTANCE_KM_STEPS = [100, 80, 60, 40, 25]
SPATIAL_CELL_SIZE_DEGREES = 0.5
MAX_EXTRACTS_PER_RUN = int(os.environ.get("WORLDWIDE_OSM_MAX_EXTRACTS_PER_RUN", "12"))
MAX_DISCOVERED_EXTRACTS = int(
    os.environ.get("WORLDWIDE_OSM_MAX_DISCOVERED_EXTRACTS", "240")
)
MAX_EXTRACT_DOWNLOAD_BYTES = int(
    os.environ.get("WORLDWIDE_OSM_MAX_EXTRACT_DOWNLOAD_BYTES", "300000000")
)
KEEP_DOWNLOADED_EXTRACTS = (
    os.environ.get("WORLDWIDE_OSM_KEEP_DOWNLOADED_EXTRACTS", "0") == "1"
)
RENDER_INTERMEDIATE_MAPS = (
    os.environ.get("WORLDWIDE_OSM_RENDER_INTERMEDIATE_MAPS", "0") == "1"
)
SAMPLE_CHECKPOINT_INTERVAL = int(
    os.environ.get("WORLDWIDE_OSM_SAMPLE_CHECKPOINT_INTERVAL", "10")
)

TOOLTIP_COLUMNS = [
    "polygon_name",
    "country",
    "world_region",
    "local_language",
    "bbox_label",
    "source_extract_id",
    "osm_type",
    "osm_id",
    "area_km2",
    "area_size_bin",
]


def load_region_sample_deficits(
    sample_path: str,
    target_sample_size: int,
) -> dict[str, int]:
    if not Path(sample_path).exists():
        return {}

    sample_gdf = load_geodataframe(sample_path)
    if "world_region" not in sample_gdf.columns:
        return {}

    return compute_region_sample_deficits(
        sample_gdf,
        target_sample_size,
        regions=configured_world_regions(),
    )


def download_json(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_text(response.text)

    return json.loads(path.read_text())


def load_geofabrik_lookup(index_path: Path) -> dict:
    index = download_json(GEOFABRIK_INDEX_URL, index_path)

    return {
        feature["properties"]["id"]: feature["properties"]
        for feature in index["features"]
    }


def _root_parent(
    extract_id: str,
    geofabrik_lookup: dict,
) -> str | None:
    current = geofabrik_lookup.get(extract_id)
    root = extract_id

    while current is not None and current.get("parent") is not None:
        root = current["parent"]
        current = geofabrik_lookup.get(root)

    return root


def _is_existing_or_parent_config(
    extract_id: str,
    configured_extract_ids: set[str],
    geofabrik_lookup: dict,
) -> bool:
    current_id = extract_id
    while current_id is not None:
        if current_id in configured_extract_ids:
            return True

        current = geofabrik_lookup.get(current_id)
        current_id = current.get("parent") if current is not None else None

    return False


def build_discovered_extract_configs(
    geofabrik_lookup: dict,
    configured_extract_ids: set[str],
) -> list[dict]:
    discovered_configs = []

    for extract_id, properties in sorted(geofabrik_lookup.items()):
        if extract_id in SKIP_DISCOVERED_EXTRACT_IDS:
            continue

        if _is_existing_or_parent_config(
            extract_id,
            configured_extract_ids,
            geofabrik_lookup,
        ):
            continue

        pbf_url = properties.get("urls", {}).get("pbf")
        if pbf_url is None:
            continue

        try:
            size = download_size(pbf_url)
        except requests.RequestException as error:
            print(f"Could not inspect discovered extract {extract_id}: {error}")
            continue

        if size is None or size > MAX_EXTRACT_DOWNLOAD_BYTES:
            continue

        root_parent = _root_parent(extract_id, geofabrik_lookup)
        world_region = REGION_BY_GEOFABRIK_ROOT.get(root_parent)
        if world_region is None:
            continue

        discovered_configs.append(
            {
                "extract_id": extract_id,
                "world_region": world_region,
                "local_language": DEFAULT_LANGUAGE_BY_REGION[world_region],
                "country": properties["name"],
            }
        )

        if len(discovered_configs) >= MAX_DISCOVERED_EXTRACTS:
            break

    return discovered_configs


def load_attempted_ids(path: str) -> set[str]:
    attempted_path = Path(path)
    if not attempted_path.exists():
        return set()

    return {
        line.strip()
        for line in attempted_path.read_text().splitlines()
        if line.strip()
    }


def save_attempted_ids(extract_ids: set[str], path: str) -> None:
    attempted_path = Path(path)
    attempted_path.parent.mkdir(parents=True, exist_ok=True)
    attempted_path.write_text("\n".join(sorted(extract_ids)) + "\n")


def download_size(url: str) -> int | None:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    content_length = response.headers.get("content-length")

    return int(content_length) if content_length is not None else None


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def remove_downloaded_extract_if_unkept(pbf_path: Path) -> None:
    if not KEEP_DOWNLOADED_EXTRACTS:
        pbf_path.unlink(missing_ok=True)


def safe_extract_filename(extract_id: str) -> str:
    return extract_id.replace("/", "__") + "-latest.osm.pbf"


def load_existing_geodataframe(path: str) -> gpd.GeoDataFrame | None:
    if not Path(path).exists():
        return None

    gdf = prepare_candidate_pool(load_geodataframe(path))

    return None if gdf.empty else gdf


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
    if "source_extract_id" in gdf.columns:
        gdf = add_extract_spatial_cells(
            gdf,
            cell_size_degrees=SPATIAL_CELL_SIZE_DEGREES,
        )
    gdf = filter_named_environmental_polygons(gdf)

    return gdf.reset_index(drop=True)


def combine_candidate_gdfs(gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    prepared_gdfs = [
        prepare_candidate_pool(gdf)
        for gdf in gdfs
        if gdf is not None and not gdf.empty
    ]
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


def process_extract(
    extract_config: dict,
    geofabrik_lookup: dict,
) -> gpd.GeoDataFrame | None:
    extract_id = extract_config["extract_id"]
    geofabrik_properties = geofabrik_lookup[extract_id]
    pbf_url = geofabrik_properties["urls"]["pbf"]
    extract_label = geofabrik_properties["name"]
    pbf_path = EXTRACT_DIR / safe_extract_filename(extract_id)

    size = download_size(pbf_url)
    if size is not None and size > MAX_EXTRACT_DOWNLOAD_BYTES:
        size_mb = size / 1_000_000
        max_mb = MAX_EXTRACT_DOWNLOAD_BYTES / 1_000_000
        print(f"Skipping {extract_id}: {size_mb:.1f} MB exceeds {max_mb:.1f} MB")
        if not KEEP_DOWNLOADED_EXTRACTS:
            pbf_path.unlink(missing_ok=True)
        return None

    print(f"Downloading {extract_id} from {pbf_url}")
    download_file(pbf_url, pbf_path)
    print(f"Reading named environmental multipolygons from {pbf_path}")
    multipolygons_gdf = read_pbf_multipolygons(pbf_path)

    if multipolygons_gdf.empty:
        remove_downloaded_extract_if_unkept(pbf_path)
        return None

    candidate_gdf = multipolygons_to_candidate_gdf(
        multipolygons_gdf,
        {
            "extract_id": extract_id,
            "extract_label": extract_label,
            "country": extract_config.get("country", extract_label),
            "world_region": extract_config["world_region"],
            "local_language": extract_config["local_language"],
        },
    )

    if candidate_gdf.empty:
        remove_downloaded_extract_if_unkept(pbf_path)
        return None

    candidate_gdf = add_geodesic_area_km2(candidate_gdf)
    candidate_gdf = filter_by_area(
        candidate_gdf,
        min_area_km2=MIN_AREA_KM2,
        max_area_km2=MAX_AREA_KM2,
    )

    if candidate_gdf.empty:
        remove_downloaded_extract_if_unkept(pbf_path)
        return None

    candidate_gdf = add_area_size_bin(candidate_gdf)
    candidate_gdf = add_extract_spatial_cells(
        candidate_gdf,
        cell_size_degrees=SPATIAL_CELL_SIZE_DEGREES,
    )
    candidate_gdf = filter_named_environmental_polygons(candidate_gdf)

    remove_downloaded_extract_if_unkept(pbf_path)

    return candidate_gdf


def sample_with_relaxed_sparsity(candidates_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    best_sample = candidates_gdf.head(0).copy()
    best_distance_km = None

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
        print(
            "Region count spread for this pass: "
            f"{region_count_spread(sample_gdf, regions=configured_world_regions())}"
        )

        if is_better_worldwide_sample(
            candidate_sample=sample_gdf,
            candidate_distance_km=min_distance_km,
            current_best_sample=best_sample,
            current_best_distance_km=best_distance_km,
            target_sample_size=SAMPLE_SIZE,
            regions=configured_world_regions(),
        ):
            best_sample = sample_gdf
            best_distance_km = min_distance_km

    print(f"Selected sparsity pass: {best_distance_km} km minimum centroid distance")

    return best_sample


def render_sample_map(sample_gdf: gpd.GeoDataFrame) -> None:
    create_polygon_map(
        sample_gdf,
        MAP_OUTPUT_PATH,
        color_by="world_region",
        tooltip_columns=TOOLTIP_COLUMNS,
        title="Worldwide OSM polygon training sample",
    )
    print(f"Created sample map at {MAP_OUTPUT_PATH}")


def save_sample_checkpoint(candidates_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sample_gdf = sample_with_relaxed_sparsity(candidates_gdf)
    save_geodataframe(sample_gdf, SAMPLE_OUTPUT_PATH)
    print(f"Saved {len(sample_gdf)} sampled polygons to {SAMPLE_OUTPUT_PATH}")

    if RENDER_INTERMEDIATE_MAPS:
        render_sample_map(sample_gdf)

    return sample_gdf


def should_update_sample_checkpoint(
    processed_extract_index: int,
    checkpoint_interval: int,
) -> bool:
    if checkpoint_interval <= 0:
        return False

    return processed_extract_index % checkpoint_interval == 0


def resample_existing_candidates() -> None:
    existing_extract_gdf = load_existing_geodataframe(EXTRACT_CANDIDATES_PATH)
    if existing_extract_gdf is None:
        raise RuntimeError(
            "No cached extract candidates found; run with "
            "WORLDWIDE_OSM_MAX_EXTRACTS_PER_RUN > 0 first."
        )

    combined_inputs = [existing_extract_gdf]
    overpass_gdf = load_existing_geodataframe(OVERPASS_CANDIDATES_PATH)
    if overpass_gdf is not None:
        combined_inputs.append(overpass_gdf)

    extract_candidates_gdf = combine_candidate_gdfs([existing_extract_gdf])
    save_geodataframe(extract_candidates_gdf, EXTRACT_CANDIDATES_PATH)
    combined_candidates_gdf = combine_candidate_gdfs(combined_inputs)
    save_geodataframe(combined_candidates_gdf, COMBINED_CANDIDATES_PATH)
    print(f"Prepared {len(combined_candidates_gdf)} cached candidate polygons")

    sample_gdf = save_sample_checkpoint(combined_candidates_gdf)
    render_sample_map(sample_gdf)

    if len(sample_gdf) < SAMPLE_SIZE:
        print(
            f"Sample is {SAMPLE_SIZE - len(sample_gdf)} polygons below the "
            f"{SAMPLE_SIZE} polygon target; rerun this script to process more extracts."
        )


def main() -> None:
    if MAX_EXTRACTS_PER_RUN == 0:
        resample_existing_candidates()
        return

    geofabrik_lookup = load_geofabrik_lookup(INDEX_PATH)
    configured_extract_ids = {
        extract_config["extract_id"]
        for extract_config in EXTRACT_CONFIGS
    }
    attempted_extract_ids = load_attempted_ids(ATTEMPTED_EXTRACT_IDS_PATH)
    configured_pending_configs = [
        extract_config
        for extract_config in EXTRACT_CONFIGS
        if extract_config["extract_id"] not in attempted_extract_ids
    ]
    if len(configured_pending_configs) >= MAX_EXTRACTS_PER_RUN:
        discovered_configs = []
    else:
        discovered_configs = build_discovered_extract_configs(
            geofabrik_lookup,
            configured_extract_ids,
        )
    all_extract_configs = EXTRACT_CONFIGS + discovered_configs
    region_deficits = load_region_sample_deficits(SAMPLE_OUTPUT_PATH, SAMPLE_SIZE)
    if region_deficits:
        print("Current sample deficit by world region:")
        print(pd.Series(region_deficits).sort_values(ascending=False))
        all_extract_configs = rank_extract_configs_by_region_deficit(
            all_extract_configs,
            region_deficits,
        )
    candidate_gdfs = []

    existing_extract_gdf = load_existing_geodataframe(EXTRACT_CANDIDATES_PATH)
    if existing_extract_gdf is not None:
        candidate_gdfs.append(existing_extract_gdf)
        print(f"Loaded {len(existing_extract_gdf)} existing extract candidates")

    extracts_to_process = [
        extract_config
        for extract_config in all_extract_configs
        if extract_config["extract_id"] not in attempted_extract_ids
    ][:MAX_EXTRACTS_PER_RUN]

    print(
        f"Configured {len(EXTRACT_CONFIGS)} curated extracts and "
        f"{len(discovered_configs)} discovered small extracts"
    )

    for index, extract_config in enumerate(extracts_to_process, start=1):
        extract_id = extract_config["extract_id"]
        print(f"Processing extract {index}/{len(extracts_to_process)}: {extract_id}")

        try:
            extract_gdf = process_extract(extract_config, geofabrik_lookup)
        except Exception as error:
            print(f"Failed extract {extract_id}: {error}")
            extract_gdf = None

        attempted_extract_ids.add(extract_id)
        save_attempted_ids(attempted_extract_ids, ATTEMPTED_EXTRACT_IDS_PATH)

        if extract_gdf is None or extract_gdf.empty:
            print("No usable polygons after extract filtering")
            continue

        print(f"Kept {len(extract_gdf)} polygons from {extract_id}")
        candidate_gdfs.append(extract_gdf)
        extract_candidates_gdf = combine_candidate_gdfs(candidate_gdfs)
        save_geodataframe(extract_candidates_gdf, EXTRACT_CANDIDATES_PATH)

        overpass_gdf = load_existing_geodataframe(OVERPASS_CANDIDATES_PATH)
        combined_inputs = [extract_candidates_gdf]
        if overpass_gdf is not None:
            combined_inputs.append(overpass_gdf)
        combined_candidates_gdf = combine_candidate_gdfs(combined_inputs)
        save_geodataframe(combined_candidates_gdf, COMBINED_CANDIDATES_PATH)
        if should_update_sample_checkpoint(index, SAMPLE_CHECKPOINT_INTERVAL):
            sample_gdf = save_sample_checkpoint(combined_candidates_gdf)

            if len(sample_gdf) >= SAMPLE_SIZE:
                render_sample_map(sample_gdf)
                print("Reached the worldwide polygon sample target")
                return

    if not candidate_gdfs:
        raise RuntimeError("No extract candidates were collected")

    extract_candidates_gdf = combine_candidate_gdfs(candidate_gdfs)
    save_geodataframe(extract_candidates_gdf, EXTRACT_CANDIDATES_PATH)
    overpass_gdf = load_existing_geodataframe(OVERPASS_CANDIDATES_PATH)
    combined_inputs = [extract_candidates_gdf]
    if overpass_gdf is not None:
        combined_inputs.append(overpass_gdf)
    combined_candidates_gdf = combine_candidate_gdfs(combined_inputs)
    save_geodataframe(combined_candidates_gdf, COMBINED_CANDIDATES_PATH)
    sample_gdf = save_sample_checkpoint(combined_candidates_gdf)
    render_sample_map(sample_gdf)

    if len(sample_gdf) < SAMPLE_SIZE:
        print(
            f"Sample is {SAMPLE_SIZE - len(sample_gdf)} polygons below the "
            f"{SAMPLE_SIZE} polygon target; rerun this script to process more extracts."
        )


if __name__ == "__main__":
    main()

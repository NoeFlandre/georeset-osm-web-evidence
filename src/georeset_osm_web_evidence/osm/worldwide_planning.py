import pandas as pd


def compute_region_sample_deficits(
    sample_df: pd.DataFrame,
    target_sample_size: int,
    regions: list[str],
) -> dict[str, int]:
    if not regions:
        return {}

    sorted_regions = sorted(regions)
    base_target = target_sample_size // len(sorted_regions)
    remainder = target_sample_size % len(sorted_regions)
    region_targets = {
        region: base_target + int(index < remainder)
        for index, region in enumerate(sorted_regions)
    }
    region_counts = sample_df["world_region"].value_counts().to_dict()

    return {
        region: max(region_targets[region] - int(region_counts.get(region, 0)), 0)
        for region in sorted_regions
    }


def rank_extract_configs_by_region_deficit(
    extract_configs: list[dict],
    region_deficits: dict[str, int],
) -> list[dict]:
    indexed_configs = list(enumerate(extract_configs))
    indexed_configs.sort(
        key=lambda item: (
            -region_deficits.get(item[1]["world_region"], 0),
            item[0],
        )
    )

    return [extract_config for _, extract_config in indexed_configs]


def region_count_spread(sample_df: pd.DataFrame, regions: list[str]) -> int:
    if sample_df.empty or "world_region" not in sample_df.columns or not regions:
        return 0

    region_counts = sample_df["world_region"].value_counts().to_dict()
    counts = [int(region_counts.get(region, 0)) for region in sorted(regions)]

    return max(counts) - min(counts)


def is_better_worldwide_sample(
    candidate_sample: pd.DataFrame,
    candidate_distance_km: int,
    current_best_sample: pd.DataFrame,
    current_best_distance_km: int | None,
    target_sample_size: int,
    regions: list[str],
) -> bool:
    candidate_is_full = len(candidate_sample) >= target_sample_size
    current_best_is_full = len(current_best_sample) >= target_sample_size

    if candidate_is_full != current_best_is_full:
        return candidate_is_full

    if not candidate_is_full and len(candidate_sample) != len(current_best_sample):
        return len(candidate_sample) > len(current_best_sample)

    candidate_spread = region_count_spread(candidate_sample, regions)
    current_best_spread = region_count_spread(current_best_sample, regions)
    if candidate_spread != current_best_spread:
        return candidate_spread < current_best_spread

    if current_best_distance_km is None:
        return True

    return candidate_distance_km > current_best_distance_km

import unittest

import pandas as pd

from georeset_osm_web_evidence.osm.worldwide_balancing import compute_group_targets


class WorldwideBalancingTests(unittest.TestCase):
    def test_compute_group_targets_prioritizes_region_before_area_bins(self) -> None:
        rows = []
        area_bins_by_region = {
            "Africa": ["small", "medium"],
            "Asia": ["small", "medium", "large", "tiny"],
            "Europe": ["small", "medium", "large", "tiny"],
        }

        for region, area_bins in area_bins_by_region.items():
            for area_bin in area_bins:
                for _ in range(6):
                    rows.append({
                        "world_region": region,
                        "area_size_bin": area_bin,
                    })

        targets = compute_group_targets(
            pd.DataFrame(rows),
            sample_size=30,
            group_columns=["world_region", "area_size_bin"],
        )

        region_targets = {}
        for group_key, target in targets.items():
            region = group_key[0]
            region_targets[region] = region_targets.get(region, 0) + target

        self.assertEqual(region_targets, {
            "Africa": 10,
            "Asia": 10,
            "Europe": 10,
        })

    def test_compute_group_targets_reallocates_when_group_is_too_small(self) -> None:
        df = pd.DataFrame(
            [{"world_region": "Africa"}]
            + [{"world_region": "Asia"} for _ in range(5)]
            + [{"world_region": "Europe"} for _ in range(5)]
        )

        targets = compute_group_targets(
            df,
            sample_size=9,
            group_columns=["world_region"],
        )

        self.assertEqual(targets, {
            ("Africa",): 1,
            ("Asia",): 4,
            ("Europe",): 4,
        })


if __name__ == "__main__":
    unittest.main()

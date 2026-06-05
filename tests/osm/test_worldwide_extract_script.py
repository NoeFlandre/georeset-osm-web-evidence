import unittest

import pandas as pd

from scripts.osm.build_worldwide_polygon_sample_from_extracts import (
    is_better_worldwide_sample,
    rank_extract_configs_by_region_deficit,
)


class WorldwideExtractScriptTests(unittest.TestCase):
    def test_rank_extract_configs_by_region_deficit_prioritizes_underrepresented_regions(
        self,
    ) -> None:
        extract_configs = [
            {"extract_id": "europe-a", "world_region": "Europe"},
            {"extract_id": "south-america-a", "world_region": "South America"},
            {"extract_id": "africa-a", "world_region": "Africa"},
            {"extract_id": "south-america-b", "world_region": "South America"},
        ]
        region_deficits = {
            "Africa": 20,
            "Europe": 0,
            "South America": 100,
        }

        ranked_configs = rank_extract_configs_by_region_deficit(
            extract_configs,
            region_deficits,
        )

        self.assertEqual(
            [config["extract_id"] for config in ranked_configs],
            ["south-america-a", "south-america-b", "africa-a", "europe-a"],
        )

    def test_is_better_worldwide_sample_prefers_region_balance_for_full_samples(
        self,
    ) -> None:
        worse_balanced_sample = pd.DataFrame(
            {
                "world_region": [
                    *["Africa"] * 5,
                    *["Asia"] * 9,
                    *["Europe"] * 4,
                ]
            }
        )
        better_balanced_sample = pd.DataFrame(
            {
                "world_region": [
                    *["Africa"] * 6,
                    *["Asia"] * 6,
                    *["Europe"] * 6,
                ]
            }
        )

        result = is_better_worldwide_sample(
            candidate_sample=better_balanced_sample,
            candidate_distance_km=25,
            current_best_sample=worse_balanced_sample,
            current_best_distance_km=40,
            target_sample_size=18,
        )

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

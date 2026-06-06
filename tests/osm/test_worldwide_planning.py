import unittest

import pandas as pd

from georeset_osm_web_evidence.osm.worldwide_planning import (
    compute_region_sample_deficits,
    is_better_worldwide_sample,
    rank_extract_configs_by_region_deficit,
)


class WorldwidePlanningTests(unittest.TestCase):
    def test_compute_region_sample_deficits_uses_explicit_region_targets(self) -> None:
        sample_df = pd.DataFrame(
            {
                "world_region": [
                    "Africa",
                    "Asia",
                    "Asia",
                    "Europe",
                ]
            }
        )

        deficits = compute_region_sample_deficits(
            sample_df,
            target_sample_size=9,
            regions=["Africa", "Asia", "Europe"],
        )

        self.assertEqual(
            deficits,
            {
                "Africa": 2,
                "Asia": 1,
                "Europe": 2,
            },
        )

    def test_rank_extract_configs_by_region_deficit_keeps_original_order_for_ties(
        self,
    ) -> None:
        extract_configs = [
            {"extract_id": "europe-a", "world_region": "Europe"},
            {"extract_id": "africa-a", "world_region": "Africa"},
            {"extract_id": "africa-b", "world_region": "Africa"},
            {"extract_id": "asia-a", "world_region": "Asia"},
        ]

        ranked_configs = rank_extract_configs_by_region_deficit(
            extract_configs,
            region_deficits={"Africa": 10, "Asia": 2, "Europe": 2},
        )

        self.assertEqual(
            [config["extract_id"] for config in ranked_configs],
            ["africa-a", "africa-b", "europe-a", "asia-a"],
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
            regions=["Africa", "Asia", "Europe"],
        )

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

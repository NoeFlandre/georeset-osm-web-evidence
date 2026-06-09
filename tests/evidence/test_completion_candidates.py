import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.completion_candidates import (
    order_completion_candidates,
    polygon_keys,
)


class CompletionCandidateTests(unittest.TestCase):
    def test_polygon_keys_returns_unique_osm_keys(self):
        df = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "name": "First"},
                {"osm_type": "way", "osm_id": 1, "name": "Duplicate"},
                {"osm_type": "relation", "osm_id": 2, "name": "Second"},
            ]
        )

        result = polygon_keys(df)

        self.assertEqual(
            result.to_dict("records"),
            [
                {"osm_type": "way", "osm_id": 1},
                {"osm_type": "relation", "osm_id": 2},
            ],
        )

    def test_orders_completion_candidates_by_underrepresented_region_and_area_bin(self):
        source_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Already Complete",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Already Attempted",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                },
                {
                    "osm_type": "way",
                    "osm_id": 3,
                    "polygon_name": "Africa Medium Reserve",
                    "world_region": "Africa",
                    "area_size_bin": "medium",
                },
                {
                    "osm_type": "way",
                    "osm_id": 4,
                    "polygon_name": "Asia Large Forest",
                    "world_region": "Asia",
                    "area_size_bin": "large",
                },
            ]
        )
        complete_df = source_df.head(1)
        attempted_df = pd.DataFrame([{"osm_type": "way", "osm_id": 2}])

        candidates = order_completion_candidates(
            source_df=source_df,
            complete_df=complete_df,
            attempted_df=attempted_df,
        )

        self.assertEqual(candidates["osm_id"].to_list(), [3, 4])


if __name__ == "__main__":
    unittest.main()

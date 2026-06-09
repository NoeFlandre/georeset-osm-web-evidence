import unittest

import pandas as pd

from georeset_osm_web_evidence.storage.dataframe import append_unique_rows


class DataFrameStorageTests(unittest.TestCase):
    def test_append_unique_rows_preserves_existing_rows_before_new_duplicates(self):
        existing_df = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "query": "forest", "url": "a"},
                {"osm_type": "way", "osm_id": 2, "query": "wetland", "url": "b"},
            ]
        )
        new_df = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 2, "query": "wetland", "url": "updated"},
                {"osm_type": "way", "osm_id": 3, "query": "forest", "url": "c"},
            ]
        )

        result = append_unique_rows(
            existing_df,
            new_df,
            subset=["osm_type", "osm_id", "query"],
        )

        self.assertEqual(
            result.to_dict("records"),
            [
                {"osm_type": "way", "osm_id": 1, "query": "forest", "url": "a"},
                {"osm_type": "way", "osm_id": 2, "query": "wetland", "url": "b"},
                {"osm_type": "way", "osm_id": 3, "query": "forest", "url": "c"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

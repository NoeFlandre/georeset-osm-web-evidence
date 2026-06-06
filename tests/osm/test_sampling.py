import unittest

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_osm_web_evidence.osm.sampling import sample_polygons


class OsmSamplingTests(unittest.TestCase):
    def test_sample_polygons_caps_requested_size(self):
        gdf = gpd.GeoDataFrame(
            [
                {"osm_id": 1, "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])},
                {"osm_id": 2, "geometry": Polygon([(2, 0), (3, 0), (3, 1), (2, 0)])},
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        sample = sample_polygons(gdf, sample_size=10, random_state=42)

        self.assertEqual(len(sample), 2)
        self.assertEqual(set(sample["osm_id"]), {1, 2})

    def test_sample_polygons_is_deterministic_for_same_seed(self):
        rows = [
            {"osm_id": index, "geometry": Polygon([(index, 0), (index + 1, 0), (index, 1), (index, 0)])}
            for index in range(10)
        ]
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample_1 = sample_polygons(gdf, sample_size=4, random_state=7)
        sample_2 = sample_polygons(gdf, sample_size=4, random_state=7)

        self.assertEqual(sample_1["osm_id"].to_list(), sample_2["osm_id"].to_list())


if __name__ == "__main__":
    unittest.main()

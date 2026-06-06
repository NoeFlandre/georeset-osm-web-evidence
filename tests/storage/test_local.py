import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_osm_web_evidence.storage.local import (
    load_geodataframe,
    save_geodataframe,
)


class LocalStorageTests(unittest.TestCase):
    def test_save_and_load_geodataframe_creates_parent_directories(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "osm_id": 1,
                    "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "sample.parquet"

            save_geodataframe(gdf, path)
            loaded = load_geodataframe(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.crs.to_string(), "EPSG:4326")
        self.assertEqual(loaded.loc[0, "osm_id"], 1)


if __name__ == "__main__":
    unittest.main()

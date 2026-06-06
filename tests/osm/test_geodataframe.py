import unittest

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_osm_web_evidence.osm.geodataframe import (
    add_area_km2,
    add_centroid_coordinates,
    add_geodesic_area_km2,
    filter_by_area,
    records_to_geodataframe,
)


class OsmGeoDataFrameTests(unittest.TestCase):
    def test_records_to_geodataframe_sets_geometry_and_crs(self):
        records = [
            {
                "osm_type": "way",
                "osm_id": 1,
                "geometry": Polygon([(1, 47), (2, 47), (2, 48), (1, 47)]),
            }
        ]

        gdf = records_to_geodataframe(records)

        self.assertEqual(gdf.crs.to_string(), "EPSG:4326")
        self.assertEqual(gdf.geometry.name, "geometry")
        self.assertEqual(gdf.loc[0, "osm_id"], 1)

    def test_filter_by_area_uses_inclusive_bounds(self):
        gdf = gpd.GeoDataFrame(
            [
                {"osm_id": 1, "area_km2": 1.0, "geometry": Polygon()},
                {"osm_id": 2, "area_km2": 5.0, "geometry": Polygon()},
                {"osm_id": 3, "area_km2": 10.0, "geometry": Polygon()},
                {"osm_id": 4, "area_km2": 11.0, "geometry": Polygon()},
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = filter_by_area(gdf, min_area_km2=1.0, max_area_km2=10.0)

        self.assertEqual(result["osm_id"].to_list(), [1, 2, 3])

    def test_add_centroid_coordinates_preserves_original_crs(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "geometry": Polygon(
                        [(1.0, 47.0), (1.2, 47.0), (1.2, 47.2), (1.0, 47.0)]
                    )
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = add_centroid_coordinates(gdf)

        self.assertEqual(result.crs.to_string(), "EPSG:4326")
        self.assertGreater(result.loc[0, "centroid_lon"], 1.0)
        self.assertLess(result.loc[0, "centroid_lon"], 1.2)
        self.assertGreater(result.loc[0, "centroid_lat"], 47.0)
        self.assertLess(result.loc[0, "centroid_lat"], 47.2)

    def test_area_helpers_add_positive_area_without_mutating_input(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "geometry": Polygon(
                        [(1.0, 47.0), (1.2, 47.0), (1.2, 47.2), (1.0, 47.0)]
                    )
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        projected_area = add_area_km2(gdf)
        geodesic_area = add_geodesic_area_km2(gdf)

        self.assertNotIn("area_km2", gdf.columns)
        self.assertGreater(projected_area.loc[0, "area_km2"], 0)
        self.assertGreater(geodesic_area.loc[0, "area_km2"], 0)


if __name__ == "__main__":
    unittest.main()

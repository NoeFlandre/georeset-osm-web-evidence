import unittest

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_osm_web_evidence.osm.extracts import (
    add_extract_spatial_cells,
    build_environmental_where_clause,
    multipolygons_to_candidate_gdf,
)


class OsmExtractTests(unittest.TestCase):
    def test_build_environmental_where_clause_requires_name_and_target_tags(self) -> None:
        where_clause = build_environmental_where_clause()

        self.assertIn("name IS NOT NULL", where_clause)
        self.assertIn("landuse IN", where_clause)
        self.assertIn("natural IN", where_clause)
        self.assertIn("boundary = 'protected_area'", where_clause)

    def test_multipolygons_to_candidate_gdf_uses_relation_or_way_ids(self) -> None:
        source_gdf = gpd.GeoDataFrame(
            [
                {
                    "osm_id": "10",
                    "osm_way_id": None,
                    "name": "Named forest",
                    "landuse": "forest",
                    "natural": None,
                    "leisure": None,
                    "boundary": None,
                    "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                },
                {
                    "osm_id": None,
                    "osm_way_id": "20",
                    "name": "Named meadow",
                    "landuse": "meadow",
                    "natural": None,
                    "leisure": None,
                    "boundary": None,
                    "geometry": Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
                },
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        extract_config = {
            "extract_id": "test-extract",
            "extract_label": "Test extract",
            "country": "Testland",
            "world_region": "Europe",
            "local_language": "en",
        }

        result = multipolygons_to_candidate_gdf(source_gdf, extract_config)

        self.assertEqual(result["osm_type"].to_list(), ["relation", "way"])
        self.assertEqual(result["osm_id"].to_list(), [10, 20])
        self.assertEqual(result["osm_tags"].iloc[0]["name"], "Named forest")
        self.assertEqual(result["source_extract_id"].to_list(), ["test-extract", "test-extract"])

    def test_add_extract_spatial_cells_assigns_global_grid_bbox_ids(self) -> None:
        gdf = gpd.GeoDataFrame(
            [
                {
                    "source_extract_id": "test-extract",
                    "geometry": Polygon([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.1)]),
                },
                {
                    "source_extract_id": "test-extract",
                    "geometry": Polygon([(1.1, 0.1), (1.2, 0.1), (1.2, 0.2), (1.1, 0.1)]),
                },
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = add_extract_spatial_cells(gdf, cell_size_degrees=1)

        self.assertEqual(
            result["bbox_id"].to_list(),
            [
                "cell:lat0_lon0",
                "cell:lat0_lon1",
            ],
        )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_osm_web_evidence.viz.map import (
    create_polygon_map,
    prepare_map_geodataframe,
    style_polygon,
)


class VizMapTests(unittest.TestCase):
    def make_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "area_km2": 12.5,
                    "has_wikipedia_articles": True,
                    "osm_tags": {"name": "Forest A", "landuse": "forest"},
                    "geometry": Polygon(
                        [
                            (1.0, 47.0),
                            (1.1, 47.0),
                            (1.1, 47.1),
                            (1.0, 47.1),
                            (1.0, 47.0),
                        ]
                    ),
                },
                {
                    "osm_type": "relation",
                    "osm_id": 2,
                    "polygon_name": "Wetland B",
                    "area_km2": 4.0,
                    "has_wikipedia_articles": False,
                    "osm_tags": {"name": "Wetland B", "natural": "wetland"},
                    "geometry": Polygon(
                        [
                            (2.0, 48.0),
                            (2.1, 48.0),
                            (2.1, 48.1),
                            (2.0, 48.1),
                            (2.0, 48.0),
                        ]
                    ),
                },
            ],
            crs="EPSG:4326",
        )

    def test_prepare_map_geodataframe_serializes_nested_properties(self) -> None:
        gdf = self.make_geodataframe()

        prepared = prepare_map_geodataframe(gdf)

        self.assertEqual(prepared.crs.to_epsg(), 4326)
        self.assertIsInstance(prepared.loc[0, "osm_tags"], str)
        self.assertIn("Forest A", prepared.loc[0, "osm_tags"])

    def test_prepare_map_geodataframe_derives_polygon_name_from_osm_tags(self) -> None:
        gdf = self.make_geodataframe().drop(columns=["polygon_name"])

        prepared = prepare_map_geodataframe(gdf)

        self.assertEqual(prepared["polygon_name"].to_list(), ["Forest A", "Wetland B"])

    def test_create_polygon_map_writes_readable_html(self) -> None:
        gdf = self.make_geodataframe()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "polygons.html"

            create_polygon_map(
                gdf,
                output_path,
                color_by="has_wikipedia_articles",
                title="Polygon audit map",
            )

            html = output_path.read_text()

        self.assertIn("Polygon audit map", html)
        self.assertIn("Forest A", html)
        self.assertIn("Wetland B", html)
        self.assertIn("Has Wikipedia articles", html)

    def test_create_polygon_map_derives_names_without_dumping_osm_tags(self) -> None:
        gdf = self.make_geodataframe().drop(columns=["polygon_name"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "polygons.html"

            create_polygon_map(gdf, output_path)

            html = output_path.read_text()

        self.assertIn("Forest A", html)
        self.assertNotIn("landuse", html)

    def test_style_polygon_treats_false_string_as_false(self) -> None:
        style = style_polygon({"properties": {"has_wikipedia_articles": "False"}})

        self.assertEqual(style["fillColor"], "#2563eb")


if __name__ == "__main__":
    unittest.main()

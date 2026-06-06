import unittest

from shapely.geometry import Polygon

from georeset_osm_web_evidence.osm.geometry import (
    element_to_polygon,
    element_to_record,
    elements_to_records,
    filter_records_with_name,
    record_has_name,
)


class OsmGeometryTests(unittest.TestCase):
    def test_converts_osm_geometry_to_polygon_lon_lat_order(self):
        element = {
            "geometry": [
                {"lat": 48.0, "lon": 2.0},
                {"lat": 48.0, "lon": 2.1},
                {"lat": 48.1, "lon": 2.1},
                {"lat": 48.0, "lon": 2.0},
            ]
        }

        polygon = element_to_polygon(element)

        self.assertIsInstance(polygon, Polygon)
        self.assertEqual(list(polygon.exterior.coords)[0], (2.0, 48.0))

    def test_rejects_missing_or_too_short_geometry(self):
        self.assertEqual(element_to_polygon({}), None)
        self.assertEqual(
            element_to_polygon(
                {
                    "geometry": [
                        {"lat": 48.0, "lon": 2.0},
                        {"lat": 48.0, "lon": 2.1},
                        {"lat": 48.1, "lon": 2.1},
                    ]
                }
            ),
            None,
        )

    def test_converts_valid_element_to_record(self):
        element = {
            "type": "way",
            "id": 123,
            "tags": {"name": "Named forest", "landuse": "forest"},
            "geometry": [
                {"lat": 48.0, "lon": 2.0},
                {"lat": 48.0, "lon": 2.1},
                {"lat": 48.1, "lon": 2.1},
                {"lat": 48.0, "lon": 2.0},
            ],
        }

        record = element_to_record(element)

        self.assertEqual(record["osm_type"], "way")
        self.assertEqual(record["osm_id"], 123)
        self.assertEqual(record["osm_tags"]["name"], "Named forest")
        self.assertIsInstance(record["geometry"], Polygon)

    def test_elements_to_records_filters_invalid_elements(self):
        elements = [
            {"type": "way", "id": 1},
            {
                "type": "way",
                "id": 2,
                "geometry": [
                    {"lat": 48.0, "lon": 2.0},
                    {"lat": 48.0, "lon": 2.1},
                    {"lat": 48.1, "lon": 2.1},
                    {"lat": 48.0, "lon": 2.0},
                ],
            },
        ]

        records = elements_to_records(elements)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["osm_id"], 2)

    def test_filters_records_with_usable_name(self):
        records = [
            {"osm_tags": {"name": "Named forest"}},
            {"osm_tags": {"name": "  "}},
            {"osm_tags": {}},
        ]

        self.assertTrue(record_has_name(records[0]))
        self.assertFalse(record_has_name(records[1]))
        self.assertEqual(filter_records_with_name(records), [records[0]])


if __name__ == "__main__":
    unittest.main()

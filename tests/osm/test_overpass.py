import unittest
from unittest.mock import Mock, patch

import requests

from georeset_osm_web_evidence.osm.overpass import (
    OVERPASS_URL,
    build_polygon_query,
    fetch_overpass_json,
)


class OsmOverpassTests(unittest.TestCase):
    def test_build_polygon_query_includes_all_tags_and_geometry_output(self):
        query = build_polygon_query(
            south=47.0,
            west=1.0,
            north=48.0,
            east=2.0,
            tags=[("landuse", "forest"), ("natural", "wetland")],
        )

        self.assertIn('way["landuse"="forest"](47.0,1.0,48.0,2.0);', query)
        self.assertIn('relation["natural"="wetland"](47.0,1.0,48.0,2.0);', query)
        self.assertIn("[out:json][timeout:240];", query)
        self.assertIn("out geom;", query)

    @patch("georeset_osm_web_evidence.osm.overpass.requests.post")
    def test_fetch_overpass_json_posts_query_with_json_headers(self, post):
        response = Mock()
        response.ok = True
        response.json.return_value = {"elements": [{"id": 1}]}
        response.raise_for_status.return_value = None
        post.return_value = response

        data = fetch_overpass_json("query", max_retries=1)

        self.assertEqual(data, {"elements": [{"id": 1}]})
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(kwargs["data"], {"data": "query"})
        self.assertEqual(kwargs["headers"]["Accept"], "application/json")
        self.assertEqual(post.call_args.args[0], OVERPASS_URL)

    @patch("georeset_osm_web_evidence.osm.overpass.time.sleep")
    @patch("georeset_osm_web_evidence.osm.overpass.requests.post")
    def test_fetch_overpass_json_retries_request_exceptions(self, post, sleep):
        response = Mock()
        response.ok = True
        response.json.return_value = {"elements": []}
        response.raise_for_status.return_value = None
        post.side_effect = [requests.Timeout("busy"), response]

        data = fetch_overpass_json(
            "query",
            max_retries=2,
            retry_delay_seconds=0,
        )

        self.assertEqual(data, {"elements": []})
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()

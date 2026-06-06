import unittest
from unittest.mock import Mock, patch

import requests

from georeset_osm_web_evidence.wikipedia.api import geosearch_wikipedia


class WikipediaApiTests(unittest.TestCase):
    @patch("georeset_osm_web_evidence.wikipedia.api.requests.get")
    def test_geosearch_wikipedia_calls_language_specific_api(self, get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "query": {
                "geosearch": [
                    {"title": "Forest", "lat": 48.0, "lon": 2.0},
                ]
            }
        }
        get.return_value = response

        result = geosearch_wikipedia(
            lat=48.0,
            lon=2.0,
            language="fr",
            radius_m=500,
            limit=3,
            max_retries=1,
        )

        self.assertEqual(result, [{"title": "Forest", "lat": 48.0, "lon": 2.0}])
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], "https://fr.wikipedia.org/w/api.php")
        _, kwargs = get.call_args
        self.assertEqual(kwargs["params"]["gscoord"], "48.0|2.0")
        self.assertEqual(kwargs["params"]["gsradius"], 500)
        self.assertEqual(kwargs["params"]["gslimit"], 3)

    @patch("georeset_osm_web_evidence.wikipedia.api.time.sleep")
    @patch("georeset_osm_web_evidence.wikipedia.api.requests.get")
    def test_geosearch_wikipedia_retries_http_errors(self, get, sleep):
        failed_response = Mock()
        failed_response.raise_for_status.side_effect = requests.HTTPError("429")
        successful_response = Mock()
        successful_response.raise_for_status.return_value = None
        successful_response.json.return_value = {"query": {"geosearch": []}}
        get.side_effect = [failed_response, successful_response]

        result = geosearch_wikipedia(
            lat=48.0,
            lon=2.0,
            language="fr",
            max_retries=2,
            retry_delay_seconds=0,
        )

        self.assertEqual(result, [])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0)

    @patch("georeset_osm_web_evidence.wikipedia.api.time.sleep")
    @patch("georeset_osm_web_evidence.wikipedia.api.requests.get")
    def test_geosearch_wikipedia_retries_request_exceptions(self, get, sleep):
        successful_response = Mock()
        successful_response.raise_for_status.return_value = None
        successful_response.json.return_value = {"query": {"geosearch": []}}
        get.side_effect = [requests.Timeout("timeout"), successful_response]

        result = geosearch_wikipedia(
            lat=48.0,
            lon=2.0,
            language="fr",
            max_retries=2,
            retry_delay_seconds=0,
        )

        self.assertEqual(result, [])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()

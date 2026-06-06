import unittest
from unittest.mock import Mock, patch

import requests

from georeset_osm_web_evidence.search.providers import (
    BRAVE_SEARCH_URL,
    normalize_brave_result,
    search_brave,
)


class SearchProviderTests(unittest.TestCase):
    def test_normalizes_brave_result_with_missing_optional_fields(self):
        result = normalize_brave_result(
            {"title": "Title", "url": "https://example.test"},
            query='"Forest" biodiversity',
        )

        self.assertEqual(
            result,
            {
                "provider": "brave",
                "query": '"Forest" biodiversity',
                "title": "Title",
                "url": "https://example.test",
                "description": None,
            },
        )

    def test_search_brave_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                search_brave("query")

    @patch("georeset_osm_web_evidence.search.providers.requests.get")
    def test_search_brave_calls_api_and_normalizes_results(self, get):
        response = Mock()
        response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.test/result",
                        "description": "Description",
                    }
                ]
            }
        }
        response.raise_for_status.return_value = None
        get.return_value = response

        results = search_brave("query", count=3, api_key="secret", max_retries=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["provider"], "brave")
        self.assertEqual(results[0]["query"], "query")
        get.assert_called_once()
        _, kwargs = get.call_args
        self.assertEqual(get.call_args.args[0], BRAVE_SEARCH_URL)
        self.assertEqual(kwargs["params"]["q"], "query")
        self.assertEqual(kwargs["params"]["count"], 3)
        self.assertEqual(kwargs["headers"]["X-Subscription-Token"], "secret")

    @patch("georeset_osm_web_evidence.search.providers.time.sleep")
    @patch("georeset_osm_web_evidence.search.providers.requests.get")
    def test_search_brave_retries_then_raises_last_request_error(self, get, sleep):
        get.side_effect = requests.Timeout("timeout")

        with self.assertRaises(requests.Timeout):
            search_brave(
                "query",
                api_key="secret",
                max_retries=2,
                retry_delay_seconds=0,
            )

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()

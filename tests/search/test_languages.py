import unittest
from collections import namedtuple

from georeset_osm_web_evidence.search.languages import resolve_query_local_language


class SearchLanguageTests(unittest.TestCase):
    def test_resolves_query_language_from_extract_override(self):
        row = {
            "source_extract_id": "extremadura",
            "local_language": "en",
        }

        self.assertEqual(resolve_query_local_language(row), "es")

    def test_resolves_query_language_from_dataframe_tuple_row(self):
        Row = namedtuple("Row", ["source_extract_id", "local_language"])
        row = Row(source_extract_id="greenland", local_language="en")

        self.assertEqual(resolve_query_local_language(row), "da")

    def test_uses_configured_language_without_override(self):
        row = {
            "source_extract_id": "sri-lanka",
            "local_language": "si",
        }

        self.assertEqual(resolve_query_local_language(row), "si")


if __name__ == "__main__":
    unittest.main()

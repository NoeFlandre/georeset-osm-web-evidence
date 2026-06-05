import unittest
from datetime import datetime
from types import SimpleNamespace

from georeset_osm_web_evidence.search.results import (
    attempt_to_row,
    result_to_row,
)


class SearchResultsTests(unittest.TestCase):
    def test_builds_search_result_row(self):
        polygon_row = SimpleNamespace(
            osm_type="way",
            osm_id=123,
            has_wikipedia_articles=True,
        )
        result = {
            "provider": "brave",
            "title": "Result title",
            "url": "https://example.com/result",
            "description": "Result description",
        }

        row = result_to_row(
            polygon_row=polygon_row,
            polygon_name="Forêt Alpha",
            query='"Forêt Alpha" forest',
            rank=2,
            result=result,
        )

        self.assertEqual(
            row,
            {
                "osm_type": "way",
                "osm_id": 123,
                "polygon_name": "Forêt Alpha",
                "has_wikipedia_articles": True,
                "query": '"Forêt Alpha" forest',
                "provider": "brave",
                "rank": 2,
                "title": "Result title",
                "url": "https://example.com/result",
                "description": "Result description",
            },
        )

    def test_builds_search_attempt_row(self):
        polygon_row = SimpleNamespace(
            osm_type="relation",
            osm_id=456,
            has_wikipedia_articles=False,
        )

        row = attempt_to_row(
            polygon_row=polygon_row,
            polygon_name="Marais Beta",
            query='"Marais Beta" wetland',
            result_count=5,
        )

        self.assertEqual(row["osm_type"], "relation")
        self.assertEqual(row["osm_id"], 456)
        self.assertEqual(row["polygon_name"], "Marais Beta")
        self.assertEqual(row["has_wikipedia_articles"], False)
        self.assertEqual(row["query"], '"Marais Beta" wetland')
        self.assertEqual(row["result_count"], 5)

        attempted_at = datetime.fromisoformat(row["attempted_at"])
        self.assertIsNotNone(attempted_at.tzinfo)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

from georeset_osm_web_evidence.evidence.page_text import (
    build_page_text_row,
    combine_queries_for_review,
)


class TestEvidencePageText(unittest.TestCase):
    def test_combines_queries_for_review(self):
        self.assertEqual(
            combine_queries_for_review(['"Forêt Alpha" forêt', '"Forêt Alpha" forest']),
            '"Forêt Alpha" forêt; "Forêt Alpha" forest',
        )
        self.assertEqual(combine_queries_for_review(None), "")

    def test_builds_page_text_row_from_candidate_url_and_fetch_result(self):
        candidate_url_row = SimpleNamespace(
            osm_type="way",
            osm_id=123,
            polygon_name="Forêt Alpha",
            has_wikipedia_articles=True,
            provider="brave",
            url="https://example.com/page",
            title="Search result title",
            description="Search result description",
            queries=['"Forêt Alpha" forêt', '"Forêt Alpha" forest'],
        )
        page_text = {
            "final_url": "https://example.com/final",
            "status_code": 200,
            "title": "Fetched page title",
            "text": "Readable page text.",
            "text_length": 19,
            "fetch_error": None,
            "extraction_method": "trafilatura",
            "extraction_error": None,
        }

        row = build_page_text_row(candidate_url_row, page_text)

        self.assertEqual(
            row,
            {
                "osm_type": "way",
                "osm_id": 123,
                "polygon_name": "Forêt Alpha",
                "has_wikipedia_articles": True,
                "provider": "brave",
                "source_url": "https://example.com/page",
                "search_title": "Search result title",
                "search_description": "Search result description",
                "search_queries": '"Forêt Alpha" forêt; "Forêt Alpha" forest',
                "final_url": "https://example.com/final",
                "status_code": 200,
                "title": "Fetched page title",
                "text": "Readable page text.",
                "text_length": 19,
                "fetch_error": None,
                "extraction_method": "trafilatura",
                "extraction_error": None,
            },
        )


if __name__ == "__main__":
    unittest.main()

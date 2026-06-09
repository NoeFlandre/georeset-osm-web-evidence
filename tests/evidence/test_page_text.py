import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

import georeset_osm_web_evidence.evidence.page_text as page_text_module
from georeset_osm_web_evidence.evidence.page_text import (
    PAGE_TEXT_COLUMNS,
    backfill_cached_page_text_metadata,
    build_page_text_row,
    combine_queries_for_review,
    fetch_candidate_pages,
    page_text_quality_artifact_is_usable,
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

    def test_backfills_missing_cached_page_text_metadata(self):
        page_text_df = pd.DataFrame(
            [
                {
                    "source_url": "https://example.com/page",
                    "search_title": pd.NA,
                    "search_description": "Existing description",
                    "search_queries": pd.NA,
                    "query_language": pd.NA,
                }
            ]
        )
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "url": "https://example.com/page",
                    "title": "Candidate title",
                    "description": "Candidate description",
                    "queries": ["query one", "query two"],
                    "query_language": "en",
                }
            ]
        )

        result, changed = backfill_cached_page_text_metadata(
            page_text_df,
            candidate_urls_df,
            metadata_columns=[
                "search_title",
                "search_description",
                "search_queries",
                "query_language",
            ],
        )

        self.assertTrue(changed)
        self.assertEqual(result.loc[0, "search_title"], "Candidate title")
        self.assertEqual(result.loc[0, "search_description"], "Existing description")
        self.assertEqual(result.loc[0, "search_queries"], "query one; query two")
        self.assertEqual(result.loc[0, "query_language"], "en")

    def test_page_text_quality_artifact_requires_quality_and_language_metadata(self):
        usable_df = pd.DataFrame(
            [
                {
                    "source_url": "https://example.org/page",
                    "quality_score": 0.8,
                    "query_language": "en",
                }
            ]
        )
        missing_quality_df = usable_df.drop(columns=["quality_score"])
        missing_language_df = usable_df.assign(query_language=pd.NA)

        self.assertTrue(page_text_quality_artifact_is_usable(usable_df))
        self.assertFalse(page_text_quality_artifact_is_usable(missing_quality_df))
        self.assertFalse(page_text_quality_artifact_is_usable(missing_language_df))
        self.assertFalse(page_text_quality_artifact_is_usable(usable_df.head(0)))

    def test_writes_page_text_artifact_with_canonical_columns(self):
        page_text_df = pd.DataFrame(
            [
                {
                    "source_url": "https://example.org/page",
                    "url": "https://example.org/page",
                    "polygon_name": "Forest A",
                }
            ],
            index=[12],
        )
        original_columns = page_text_df.columns.to_list()

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "page_text.parquet"

            self.assertTrue(hasattr(page_text_module, "write_page_text_artifact"))
            result = page_text_module.write_page_text_artifact(
                page_text_df,
                output_path,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(page_text_df.columns.to_list(), original_columns)
        self.assertEqual(result.columns.to_list(), PAGE_TEXT_COLUMNS)
        self.assertEqual(saved_df.columns.to_list(), PAGE_TEXT_COLUMNS)
        self.assertEqual(saved_df.loc[0, "source_url"], "https://example.org/page")
        self.assertNotIn("index", saved_df.columns)

    def test_fetch_candidate_pages_checkpoint_new_rows(self):
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forest A",
                    "has_wikipedia_articles": None,
                    "provider": "brave",
                    "url": "https://example.org/a",
                    "title": "A",
                    "description": "desc A",
                    "queries": ["q1"],
                    "best_rank": 1,
                    "world_region": "Europe",
                    "country": "France",
                    "local_language": "fr",
                    "query_local_language": "fr",
                    "query_language": "en",
                    "area_size_bin": "medium",
                    "polygon_category": "forest",
                }
            ]
        )

        def fake_fetch_page_text(url: str, timeout_seconds: int) -> dict:
            return {
                "url": url,
                "final_url": url,
                "status_code": 200,
                "title": "Fetched A",
                "text": "Readable page text.",
                "text_length": 19,
                "fetch_error": None,
                "extraction_method": "trafilatura",
                "extraction_error": None,
            }

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "page_text.parquet"
            result, changed = fetch_candidate_pages(
                candidate_urls_df,
                output_path=output_path,
                logger=None,
                fetch_page_text_func=fake_fetch_page_text,
                fetch_timeout_seconds=7,
                fetch_delay_seconds=0,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertTrue(changed)
        self.assertEqual(result.columns.to_list(), PAGE_TEXT_COLUMNS)
        self.assertEqual(result.loc[0, "source_url"], "https://example.org/a")
        self.assertEqual(result.loc[0, "query_language"], "en")
        self.assertEqual(saved_df.loc[0, "source_url"], "https://example.org/a")


if __name__ == "__main__":
    unittest.main()

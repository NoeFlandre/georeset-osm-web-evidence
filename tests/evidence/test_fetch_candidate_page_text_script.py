import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import scripts.evidence.fetch_candidate_page_text as fetch_script


class FetchCandidatePageTextScriptTests(unittest.TestCase):
    def test_fetches_candidate_page_text_with_injected_fetcher(self):
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Bois Alpha",
                    "has_wikipedia_articles": False,
                    "provider": "brave",
                    "url": "https://example.com/a",
                    "title": "Search title A",
                    "description": "Search description A",
                    "queries": ["query A"],
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Marais Beta",
                    "has_wikipedia_articles": True,
                    "provider": "brave",
                    "url": "https://example.com/b",
                    "title": "Search title B",
                    "description": "Search description B",
                    "queries": ["query B"],
                },
            ]
        )
        fetched_urls = []
        sleep_calls = []

        def fake_fetch_page_text(url: str) -> dict:
            fetched_urls.append(url)
            return {
                "url": url,
                "final_url": url,
                "status_code": 200,
                "title": f"Fetched {url}",
                "text": "Readable text.",
                "text_length": 14,
                "fetch_error": None,
                "extraction_method": "trafilatura",
                "extraction_error": None,
            }

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "candidate_urls.parquet"
            output_path = temp_path / "nested" / "page_text.parquet"
            candidate_urls_df.to_parquet(input_path, index=False)

            self.assertTrue(hasattr(fetch_script, "run_candidate_page_text_fetch"))
            result = fetch_script.run_candidate_page_text_fetch(
                input_path=input_path,
                output_path=output_path,
                url_limit=1,
                request_delay_seconds=0.25,
                fetch_page_text_func=fake_fetch_page_text,
                sleep_func=sleep_calls.append,
                print_progress=False,
            )

            saved_df = pd.read_parquet(output_path)

        self.assertEqual(fetched_urls, ["https://example.com/a"])
        self.assertEqual(sleep_calls, [0.25])
        self.assertEqual(len(result), 1)
        self.assertEqual(saved_df.loc[0, "polygon_name"], "Bois Alpha")
        self.assertEqual(saved_df.loc[0, "source_url"], "https://example.com/a")
        self.assertEqual(saved_df.loc[0, "search_queries"], "query A")


if __name__ == "__main__":
    unittest.main()

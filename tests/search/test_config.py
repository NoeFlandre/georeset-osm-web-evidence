import unittest
from pathlib import Path

from georeset_osm_web_evidence.search.config import (
    BALANCED_POLYGONS_PATH,
    BRAVE_ATTEMPTS_PATH,
    BRAVE_CANDIDATE_URLS_PATH,
    BRAVE_RESULTS_PATH,
    SEARCH_LANGUAGES,
)


class SearchConfigTests(unittest.TestCase):
    def test_exposes_shared_search_languages(self):
        self.assertEqual(SEARCH_LANGUAGES, ("fr", "en"))

    def test_exposes_shared_search_data_paths(self):
        self.assertEqual(
            BALANCED_POLYGONS_PATH,
            Path("data/processed/samples/balanced_wikipedia_100.parquet"),
        )
        self.assertEqual(
            BRAVE_RESULTS_PATH,
            Path("data/processed/search/brave_results_sample.parquet"),
        )
        self.assertEqual(
            BRAVE_ATTEMPTS_PATH,
            Path("data/processed/search/brave_search_attempts.parquet"),
        )
        self.assertEqual(
            BRAVE_CANDIDATE_URLS_PATH,
            Path("data/processed/search/brave_candidate_urls_sample.parquet"),
        )


if __name__ == "__main__":
    unittest.main()

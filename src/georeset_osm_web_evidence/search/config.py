from pathlib import Path

SEARCH_LANGUAGES = ("fr", "en")

BALANCED_POLYGONS_PATH = Path("data/processed/samples/balanced_wikipedia_100.parquet")
BRAVE_RESULTS_PATH = Path("data/processed/search/brave_results_sample.parquet")
BRAVE_ATTEMPTS_PATH = Path("data/processed/search/brave_search_attempts.parquet")
BRAVE_CANDIDATE_URLS_PATH = Path(
    "data/processed/search/brave_candidate_urls_sample.parquet"
)

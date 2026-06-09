import pandas as pd

from georeset_osm_web_evidence.search.config import (
    BRAVE_CANDIDATE_URLS_PATH,
    BRAVE_RESULTS_PATH,
)
from georeset_osm_web_evidence.search.results import (
    is_wikipedia_url,
    prepare_candidate_urls,
)
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact


def main() -> None:
    results_df = pd.read_parquet(BRAVE_RESULTS_PATH)
    raw_result_count = len(results_df)
    results_df = results_df[~results_df["url"].apply(is_wikipedia_url)].copy()
    candidate_urls_df = prepare_candidate_urls(results_df)

    output_path = BRAVE_CANDIDATE_URLS_PATH
    write_dataframe_artifact(candidate_urls_df, output_path)

    print(f"Loaded {raw_result_count} search result rows from {BRAVE_RESULTS_PATH}")
    print(f"Kept {len(results_df)} rows after removing Wikipedia URLs")
    print(f"Saved {len(candidate_urls_df)} candidate URLs to {output_path}")
    print(candidate_urls_df[["polygon_name", "best_rank", "url"]].head(20))


if __name__ == "__main__":
    main()

import pandas as pd

from georeset_osm_web_evidence.search.config import (
    BRAVE_CANDIDATE_URLS_PATH,
    BRAVE_RESULTS_PATH,
)


def is_wikipedia_url(url: str) -> bool:
    return "wikipedia.org" in url.lower()


def combine_unique_values(values) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def main() -> None:
    results_df = pd.read_parquet(BRAVE_RESULTS_PATH)
    raw_result_count = len(results_df)
    results_df = results_df[~results_df["url"].apply(is_wikipedia_url)].copy()

    candidate_urls_df = (
        results_df.sort_values("rank")
        .groupby(
            [
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "provider",
                "url",
            ],
            as_index=False,
        )
        .agg(
            best_rank=("rank", "min"),
            title=("title", "first"),
            description=("description", "first"),
            queries=("query", combine_unique_values),
        )
    )

    output_path = BRAVE_CANDIDATE_URLS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_urls_df.to_parquet(output_path, index=False)

    print(f"Loaded {raw_result_count} search result rows from {BRAVE_RESULTS_PATH}")
    print(f"Kept {len(results_df)} rows after removing Wikipedia URLs")
    print(f"Saved {len(candidate_urls_df)} candidate URLs to {output_path}")
    print(candidate_urls_df[["polygon_name", "best_rank", "url"]].head(20))


if __name__ == "__main__":
    main()

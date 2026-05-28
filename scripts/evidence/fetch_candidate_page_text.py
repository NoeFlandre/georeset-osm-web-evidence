from pathlib import Path
import time

import pandas as pd

from georeset_osm_web_evidence.web.text import fetch_page_text


def combine_queries_for_review(queries) -> str:
    if queries is None:
        return ""

    return "; ".join(str(query) for query in queries)


def main() -> None:
    input_path = "data/processed/search/brave_candidate_urls_sample.parquet"
    output_path = "data/processed/evidence/page_text_sample.parquet"
    url_limit = None
    request_delay_seconds = 1.0

    candidate_urls_df = pd.read_parquet(input_path)

    if url_limit is not None:
        candidate_urls_df = candidate_urls_df.head(url_limit)

    rows = []

    for index, row in enumerate(candidate_urls_df.itertuples(), start=1):
        print(f"Fetching URL {index}/{len(candidate_urls_df)}: {row.url}")

        page_text = fetch_page_text(row.url)

        rows.append(
            {
                "osm_type": row.osm_type,
                "osm_id": row.osm_id,
                "polygon_name": row.polygon_name,
                "has_wikipedia_articles": row.has_wikipedia_articles,
                "provider": row.provider,
                "source_url": row.url,
                "search_title": row.title,
                "search_description": row.description,
                "search_queries": combine_queries_for_review(row.queries),
                **page_text,
            }
        )

        time.sleep(request_delay_seconds)

    page_text_df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_text_df.to_parquet(output_path, index=False)

    print(f"Saved {len(page_text_df)} fetched pages to {output_path}")
    print(page_text_df[["polygon_name", "status_code", "text_length", "fetch_error"]])


if __name__ == "__main__":
    main()

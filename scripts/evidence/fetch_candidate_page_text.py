import time
from pathlib import Path
from typing import Callable

import pandas as pd

from georeset_osm_web_evidence.evidence.page_text import build_page_text_row
from georeset_osm_web_evidence.search.config import BRAVE_CANDIDATE_URLS_PATH
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact
from georeset_osm_web_evidence.web.text import fetch_page_text

DEFAULT_OUTPUT_PATH = Path("data/processed/evidence/page_text_sample.parquet")


def run_candidate_page_text_fetch(
    input_path: str | Path = BRAVE_CANDIDATE_URLS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    url_limit: int | None = None,
    request_delay_seconds: float = 1.0,
    fetch_page_text_func: Callable[[str], dict] = fetch_page_text,
    sleep_func: Callable[[float], None] = time.sleep,
    print_progress: bool = True,
) -> pd.DataFrame:
    candidate_urls_df = pd.read_parquet(input_path)

    if url_limit is not None:
        candidate_urls_df = candidate_urls_df.head(url_limit)

    rows = []

    for index, row in enumerate(candidate_urls_df.itertuples(), start=1):
        if print_progress:
            print(f"Fetching URL {index}/{len(candidate_urls_df)}: {row.url}")

        page_text = fetch_page_text_func(row.url)
        rows.append(build_page_text_row(row, page_text))

        sleep_func(request_delay_seconds)

    page_text_df = pd.DataFrame(rows)
    return write_dataframe_artifact(page_text_df, output_path)


def main() -> None:
    page_text_df = run_candidate_page_text_fetch(
        input_path=BRAVE_CANDIDATE_URLS_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        url_limit=None,
        request_delay_seconds=1.0,
    )

    print(f"Saved {len(page_text_df)} fetched pages to {DEFAULT_OUTPUT_PATH}")
    print(page_text_df[["polygon_name", "status_code", "text_length", "fetch_error"]])


if __name__ == "__main__":
    main()

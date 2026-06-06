import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from georeset_osm_web_evidence.evidence.page_text import build_page_text_row
from georeset_osm_web_evidence.evidence.sentence_candidates import (
    build_sentence_candidate_dataframe,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    add_pilot_metadata,
    attach_polygon_metadata,
    build_candidate_urls,
    build_limited_localized_queries,
    build_search_rows_for_query,
    select_stratified_pilot_polygons,
    summarize_sentence_pilot,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.terms import TERMS_BY_LANGUAGE
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.web.quality import add_quality_metadata
from georeset_osm_web_evidence.web.text import fetch_page_text


INPUT_POLYGONS_PATH = Path(
    "data/processed/samples/worldwide_training_polygon_sample.parquet"
)
OUTPUT_DIR = Path("data/processed/pilots/worldwide_sentence_pilot_10")

PILOT_POLYGONS_PATH = OUTPUT_DIR / "pilot_polygons.parquet"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "search_results.parquet"
SEARCH_ATTEMPTS_PATH = OUTPUT_DIR / "search_attempts.parquet"
CANDIDATE_URLS_PATH = OUTPUT_DIR / "candidate_urls.parquet"
FETCH_URLS_PATH = OUTPUT_DIR / "candidate_urls_to_fetch.parquet"
PAGE_TEXT_PATH = OUTPUT_DIR / "page_text.parquet"
PAGE_TEXT_WITH_QUALITY_PATH = OUTPUT_DIR / "page_text_with_quality.parquet"
SENTENCE_CANDIDATES_PATH = OUTPUT_DIR / "sentence_candidates.parquet"
ANALYSIS_PATH = OUTPUT_DIR / "analysis.json"
LOG_PATH = OUTPUT_DIR / "run.log"
RESET_OUTPUTS = os.environ.get("WORLDWIDE_SENTENCE_PILOT_RESET_OUTPUTS", "0") == "1"
OUTPUT_ARTIFACT_PATHS = [
    PILOT_POLYGONS_PATH,
    SEARCH_RESULTS_PATH,
    SEARCH_ATTEMPTS_PATH,
    CANDIDATE_URLS_PATH,
    FETCH_URLS_PATH,
    PAGE_TEXT_PATH,
    PAGE_TEXT_WITH_QUALITY_PATH,
    SENTENCE_CANDIDATES_PATH,
    ANALYSIS_PATH,
]
PILOT_REQUIRED_COLUMNS = [
    "polygon_name",
    "polygon_category",
    "query_local_language",
    "has_wikipedia_articles",
]

SAMPLE_SIZE = int(os.environ.get("WORLDWIDE_SENTENCE_PILOT_SAMPLE_SIZE", "10"))
RESULTS_PER_QUERY = int(os.environ.get("WORLDWIDE_SENTENCE_PILOT_RESULTS_PER_QUERY", "5"))
MAX_QUERIES_PER_POLYGON = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_MAX_QUERIES_PER_POLYGON", "4")
)
MAX_URLS_PER_POLYGON = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_MAX_URLS_PER_POLYGON", "3")
)
SEARCH_DELAY_SECONDS = float(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_SEARCH_DELAY_SECONDS", "1.2")
)
FETCH_DELAY_SECONDS = float(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_FETCH_DELAY_SECONDS", "1.0")
)
PAGE_TEXT_COLUMNS = [
    "osm_type",
    "osm_id",
    "polygon_name",
    "has_wikipedia_articles",
    "provider",
    "source_url",
    "search_title",
    "search_description",
    "search_queries",
    "url",
    "final_url",
    "status_code",
    "title",
    "text",
    "text_length",
    "fetch_error",
    "extraction_method",
    "extraction_error",
    "best_rank",
    "world_region",
    "country",
    "local_language",
    "query_local_language",
    "area_size_bin",
    "polygon_category",
]


def configure_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("worldwide_sentence_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def reset_output_artifacts() -> None:
    for path in OUTPUT_ARTIFACT_PATHS:
        path.unlink(missing_ok=True)


def load_or_build_dataframe(
    path: Path,
    stage_name: str,
    logger: logging.Logger,
    build: Callable[[], pd.DataFrame],
    reset: bool = False,
    load: Callable[[Path], pd.DataFrame] = pd.read_parquet,
    save: Callable[[pd.DataFrame, Path], None] | None = None,
) -> pd.DataFrame:
    if path.exists() and not reset:
        dataframe = load(path)
        logger.info("Loaded %s rows for %s from %s", len(dataframe), stage_name, path)
        return dataframe

    dataframe = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    if save is None:
        dataframe.to_parquet(path, index=False)
    else:
        save(dataframe, path)
    logger.info("Saved %s rows for %s to %s", len(dataframe), stage_name, path)

    return dataframe


def pilot_artifact_is_usable(pilot_gdf: pd.DataFrame) -> bool:
    return all(column in pilot_gdf.columns for column in PILOT_REQUIRED_COLUMNS)


def build_pilot_polygons(source_gdf: pd.DataFrame) -> pd.DataFrame:
    pilot_gdf = select_stratified_pilot_polygons(
        source_gdf,
        sample_size=SAMPLE_SIZE,
        random_state=42,
    )
    return add_pilot_metadata(pilot_gdf)


def load_or_build_pilot_polygons(
    source_gdf: pd.DataFrame,
    logger: logging.Logger,
    reset: bool,
) -> tuple[pd.DataFrame, bool]:
    if PILOT_POLYGONS_PATH.exists() and not reset:
        pilot_gdf = load_geodataframe(PILOT_POLYGONS_PATH)
        if pilot_artifact_is_usable(pilot_gdf):
            logger.info(
                "Loaded %s pilot polygons from %s",
                len(pilot_gdf),
                PILOT_POLYGONS_PATH,
            )
            return pilot_gdf, False

        logger.info(
            "Rebuilding pilot polygons because %s lacks required metadata columns",
            PILOT_POLYGONS_PATH,
        )

    pilot_gdf = build_pilot_polygons(source_gdf)
    save_geodataframe(pilot_gdf, PILOT_POLYGONS_PATH)
    logger.info("Saved %s pilot polygons to %s", len(pilot_gdf), PILOT_POLYGONS_PATH)

    return pilot_gdf, True


def build_pilot_queries(polygon_row) -> list[tuple[str, str]]:
    return build_limited_localized_queries(
        osm_tags=polygon_row.osm_tags,
        local_language=getattr(polygon_row, "query_local_language", None),
        supported_languages=set(TERMS_BY_LANGUAGE),
        max_queries=MAX_QUERIES_PER_POLYGON,
    )


def expected_query_keys(pilot_gdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for polygon_row in pilot_gdf.itertuples():
        for query_language, query in build_pilot_queries(polygon_row):
            rows.append(
                {
                    "osm_type": polygon_row.osm_type,
                    "osm_id": polygon_row.osm_id,
                    "query_language": query_language,
                    "query": query,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["osm_type", "osm_id", "query_language", "query"],
    )


def search_attempts_cover_expected_queries(
    search_attempts_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> bool:
    key_columns = ["osm_type", "osm_id", "query_language", "query"]
    if any(column not in search_attempts_df.columns for column in key_columns):
        return False

    expected_df = expected_query_keys(pilot_gdf)
    completed_df = search_attempts_df[key_columns].drop_duplicates()
    missing_df = expected_df.merge(
        completed_df,
        on=key_columns,
        how="left",
        indicator=True,
    ).query("_merge == 'left_only'")

    return missing_df.empty


def collect_search_results(
    pilot_gdf: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    attempt_rows = []

    for polygon_index, polygon_row in enumerate(pilot_gdf.itertuples(), start=1):
        queries_with_language = build_pilot_queries(polygon_row)
        logger.info(
            "Searching polygon %s/%s: %s (%s queries)",
            polygon_index,
            len(pilot_gdf),
            polygon_row.polygon_name,
            len(queries_with_language),
        )

        for query_language, query in queries_with_language:
            try:
                results = search_brave(query, count=RESULTS_PER_QUERY)
                search_error = None
            except requests.RequestException as error:
                logger.warning("Search request failed for query: %s", query)
                results = []
                search_error = str(error)

            query_result_rows, attempt_row = build_search_rows_for_query(
                polygon_row=polygon_row,
                query_language=query_language,
                query=query,
                results=results,
                search_error=search_error,
            )
            attempt_rows.append(attempt_row)
            result_rows.extend(query_result_rows)

            time.sleep(SEARCH_DELAY_SECONDS)

    return pd.DataFrame(result_rows), pd.DataFrame(attempt_rows)


def load_or_collect_search_results(
    pilot_gdf: pd.DataFrame,
    logger: logging.Logger,
    reset: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if SEARCH_RESULTS_PATH.exists() and SEARCH_ATTEMPTS_PATH.exists() and not reset:
        search_results_df = pd.read_parquet(SEARCH_RESULTS_PATH)
        search_attempts_df = pd.read_parquet(SEARCH_ATTEMPTS_PATH)
        if search_attempts_cover_expected_queries(search_attempts_df, pilot_gdf):
            logger.info(
                "Loaded %s search results and %s search attempts",
                len(search_results_df),
                len(search_attempts_df),
            )
            return search_results_df, search_attempts_df, False

        logger.info("Rebuilding search artifacts because query coverage is incomplete")

    search_results_df, search_attempts_df = collect_search_results(pilot_gdf, logger)
    search_results_df.to_parquet(SEARCH_RESULTS_PATH, index=False)
    search_attempts_df.to_parquet(SEARCH_ATTEMPTS_PATH, index=False)
    logger.info(
        "Saved %s search results to %s",
        len(search_results_df),
        SEARCH_RESULTS_PATH,
    )
    logger.info(
        "Saved %s search attempts to %s",
        len(search_attempts_df),
        SEARCH_ATTEMPTS_PATH,
    )

    return search_results_df, search_attempts_df, True


def build_candidate_url_artifacts(
    search_results_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_urls_df = build_candidate_urls(search_results_df)
    candidate_urls_df = attach_polygon_metadata(candidate_urls_df, pilot_gdf)
    fetch_urls_df = build_candidate_urls(
        search_results_df,
        max_urls_per_polygon=MAX_URLS_PER_POLYGON,
    )
    fetch_urls_df = attach_polygon_metadata(fetch_urls_df, pilot_gdf)

    return candidate_urls_df, fetch_urls_df


def load_or_build_candidate_url_artifacts(
    search_results_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
    logger: logging.Logger,
    reset: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if CANDIDATE_URLS_PATH.exists() and FETCH_URLS_PATH.exists() and not reset:
        candidate_urls_df = pd.read_parquet(CANDIDATE_URLS_PATH)
        fetch_urls_df = pd.read_parquet(FETCH_URLS_PATH)
        logger.info(
            "Loaded %s candidate URLs and %s fetch URLs",
            len(candidate_urls_df),
            len(fetch_urls_df),
        )
        return candidate_urls_df, fetch_urls_df, False

    candidate_urls_df, fetch_urls_df = build_candidate_url_artifacts(
        search_results_df,
        pilot_gdf,
    )
    candidate_urls_df.to_parquet(CANDIDATE_URLS_PATH, index=False)
    fetch_urls_df.to_parquet(FETCH_URLS_PATH, index=False)
    logger.info(
        "Saved %s candidate URLs to %s",
        len(candidate_urls_df),
        CANDIDATE_URLS_PATH,
    )
    logger.info("Selected %s URLs to fetch at %s", len(fetch_urls_df), FETCH_URLS_PATH)

    return candidate_urls_df, fetch_urls_df, True


def fetch_candidate_pages(
    candidate_urls_df: pd.DataFrame,
    logger: logging.Logger,
    output_path: Path = PAGE_TEXT_PATH,
    reset: bool = False,
) -> tuple[pd.DataFrame, bool]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not reset:
        page_text_df = pd.read_parquet(output_path)
        if "source_url" not in page_text_df.columns:
            raise ValueError(f"{output_path} is missing required column: source_url")
    else:
        page_text_df = pd.DataFrame(columns=PAGE_TEXT_COLUMNS)

    changed = False
    fetched_urls = set(page_text_df["source_url"].dropna())

    for url_index, row in enumerate(candidate_urls_df.itertuples(), start=1):
        if row.url in fetched_urls:
            logger.info(
                "Skipping already fetched URL %s/%s: %s",
                url_index,
                len(candidate_urls_df),
                row.url,
            )
            continue

        logger.info("Fetching URL %s/%s: %s", url_index, len(candidate_urls_df), row.url)
        page_text = fetch_page_text(row.url)
        page_row = build_page_text_row(row, page_text)
        page_row.update(
            {
                "best_rank": row.best_rank,
                "world_region": row.world_region,
                "country": row.country,
                "local_language": row.local_language,
                "query_local_language": row.query_local_language,
                "area_size_bin": row.area_size_bin,
                "polygon_category": row.polygon_category,
            }
        )
        page_text_df = pd.concat(
            [page_text_df, pd.DataFrame([page_row], columns=PAGE_TEXT_COLUMNS)],
            ignore_index=True,
        )
        page_text_df.to_parquet(output_path, index=False)
        fetched_urls.add(row.url)
        changed = True

        time.sleep(FETCH_DELAY_SECONDS)

    return page_text_df, changed


def enrich_sentence_metadata(
    sentence_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> pd.DataFrame:
    if sentence_df.empty:
        return sentence_df

    return attach_polygon_metadata(sentence_df, pilot_gdf)


def write_analysis(
    analysis: dict,
    logger: logging.Logger,
) -> None:
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    logger.info("Analysis: %s", json.dumps(analysis, sort_keys=True))


def main() -> None:
    logger = configure_logging()
    if RESET_OUTPUTS:
        reset_output_artifacts()

    logger.info("Starting worldwide sentence pilot")
    logger.info("Input polygons: %s", INPUT_POLYGONS_PATH)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info(
        "Pilot settings: sample_size=%s, results_per_query=%s, "
        "max_queries_per_polygon=%s, max_urls_per_polygon=%s",
        SAMPLE_SIZE,
        RESULTS_PER_QUERY,
        MAX_QUERIES_PER_POLYGON,
        MAX_URLS_PER_POLYGON,
    )

    source_gdf = load_geodataframe(INPUT_POLYGONS_PATH)
    pilot_gdf, pilot_rebuilt = load_or_build_pilot_polygons(
        source_gdf,
        logger,
        reset=RESET_OUTPUTS,
    )
    stage_reset = RESET_OUTPUTS or pilot_rebuilt
    logger.info(
        "Pilot region distribution: %s",
        pilot_gdf["world_region"].value_counts().sort_index().to_dict(),
    )
    logger.info(
        "Pilot area-bin distribution: %s",
        pilot_gdf["area_size_bin"].value_counts().sort_index().to_dict(),
    )

    search_results_df, search_attempts_df, search_rebuilt = load_or_collect_search_results(
        pilot_gdf,
        logger,
        reset=stage_reset,
    )
    candidate_reset = stage_reset or search_rebuilt
    (
        candidate_urls_df,
        fetch_urls_df,
        candidate_rebuilt,
    ) = load_or_build_candidate_url_artifacts(
        search_results_df,
        pilot_gdf,
        logger,
        reset=candidate_reset,
    )

    page_text_reset = candidate_reset or candidate_rebuilt
    page_text_df, page_text_changed = fetch_candidate_pages(
        fetch_urls_df,
        logger,
        output_path=PAGE_TEXT_PATH,
        reset=page_text_reset,
    )
    logger.info("Saved %s fetched page rows to %s", len(page_text_df), PAGE_TEXT_PATH)

    text_metadata_reset = page_text_reset or page_text_changed
    page_text_with_quality_df = load_or_build_dataframe(
        path=PAGE_TEXT_WITH_QUALITY_PATH,
        stage_name="page-text quality metadata",
        logger=logger,
        build=lambda: add_quality_metadata(page_text_df),
        reset=text_metadata_reset,
    )

    sentence_df = load_or_build_dataframe(
        path=SENTENCE_CANDIDATES_PATH,
        stage_name="sentence candidates",
        logger=logger,
        build=lambda: enrich_sentence_metadata(
            build_sentence_candidate_dataframe(page_text_with_quality_df),
            pilot_gdf,
        ),
        reset=text_metadata_reset,
    )

    analysis = summarize_sentence_pilot(
        polygons_df=pilot_gdf,
        search_results_df=search_results_df,
        candidate_urls_df=candidate_urls_df,
        page_text_df=page_text_with_quality_df,
        sentence_df=sentence_df,
    )
    write_analysis(analysis, logger)
    logger.info("Worldwide sentence pilot finished")


if __name__ == "__main__":
    main()

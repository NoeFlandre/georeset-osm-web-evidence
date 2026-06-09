import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from georeset_osm_web_evidence.evidence.page_text import (
    fetch_candidate_pages,
    page_text_quality_artifact_is_usable,
)
from georeset_osm_web_evidence.evidence.sentence_candidates import (
    MINHASH_DUPLICATE_THRESHOLD,
    build_sentence_candidate_dataframe,
    deduplicate_near_duplicate_sentence_candidates,
    select_complete_sentence_candidates,
    sentence_artifact_respects_sampling_limits,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    add_pilot_metadata,
    attach_polygon_metadata,
    build_candidate_urls,
    build_limited_localized_queries,
    build_search_rows_for_query,
    candidate_url_artifact_is_usable,
    fetch_url_artifact_matches_candidate_limit,
    filter_to_sentence_polygons,
    select_stratified_pilot_polygons,
    summarize_sentence_pilot,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.terms import TERMS_BY_LANGUAGE
from georeset_osm_web_evidence.pipeline.logging import configure_stage_logger
from georeset_osm_web_evidence.storage.dataframe import load_or_build_dataframe
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.text.sentences import (
    SENTENCE_FILTER_PROFILE,
    SENTENCE_FILTER_RULES,
)
from georeset_osm_web_evidence.web.quality import add_quality_metadata


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
COMPLETE_POLYGONS_PATH = OUTPUT_DIR / "complete_sentence_polygons.parquet"
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
    COMPLETE_POLYGONS_PATH,
    ANALYSIS_PATH,
]
PILOT_REQUIRED_COLUMNS = [
    "polygon_name",
    "polygon_category",
    "query_local_language",
    "has_wikipedia_articles",
]

TARGET_COMPLETE_POLYGON_COUNT = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_TARGET_COMPLETE_POLYGONS", "10")
)
SAMPLE_SIZE = int(
    os.environ.get(
        "WORLDWIDE_SENTENCE_PILOT_SAMPLE_SIZE",
        str(TARGET_COMPLETE_POLYGON_COUNT * 4),
    )
)
RESULTS_PER_QUERY = int(os.environ.get("WORLDWIDE_SENTENCE_PILOT_RESULTS_PER_QUERY", "5"))
MAX_QUERIES_PER_POLYGON = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_MAX_QUERIES_PER_POLYGON", "4")
)
MAX_URLS_PER_POLYGON = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_MAX_URLS_PER_POLYGON", "25")
)
MAX_SENTENCES_PER_POLYGON = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_MAX_SENTENCES_PER_POLYGON", "10")
)
MAX_SENTENCES_PER_URL = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_MAX_SENTENCES_PER_URL", "1")
)
SEARCH_DELAY_SECONDS = float(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_SEARCH_DELAY_SECONDS", "1.2")
)
FETCH_DELAY_SECONDS = float(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_FETCH_DELAY_SECONDS", "1.0")
)
FETCH_TIMEOUT_SECONDS = int(
    os.environ.get("WORLDWIDE_SENTENCE_PILOT_FETCH_TIMEOUT_SECONDS", "10")
)


def configure_logging() -> logging.Logger:
    return configure_stage_logger("worldwide_sentence_pilot", LOG_PATH)


def reset_output_artifacts() -> None:
    for path in OUTPUT_ARTIFACT_PATHS:
        path.unlink(missing_ok=True)


def load_or_build_page_text_quality(
    page_text_df: pd.DataFrame,
    logger: logging.Logger,
    reset: bool,
) -> pd.DataFrame:
    if PAGE_TEXT_WITH_QUALITY_PATH.exists() and not reset:
        page_text_with_quality_df = pd.read_parquet(PAGE_TEXT_WITH_QUALITY_PATH)
        if page_text_quality_artifact_is_usable(page_text_with_quality_df):
            logger.info(
                "Loaded %s rows for page-text quality metadata from %s",
                len(page_text_with_quality_df),
                PAGE_TEXT_WITH_QUALITY_PATH,
            )
            return page_text_with_quality_df

        logger.info(
            "Rebuilding page-text quality metadata because metadata is incomplete"
        )

    page_text_with_quality_df = add_quality_metadata(page_text_df)
    page_text_with_quality_df.to_parquet(PAGE_TEXT_WITH_QUALITY_PATH, index=False)
    logger.info(
        "Saved %s rows for page-text quality metadata to %s",
        len(page_text_with_quality_df),
        PAGE_TEXT_WITH_QUALITY_PATH,
    )
    return page_text_with_quality_df


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
        if (
            candidate_url_artifact_is_usable(candidate_urls_df)
            and candidate_url_artifact_is_usable(fetch_urls_df)
            and fetch_url_artifact_matches_candidate_limit(
                candidate_urls_df,
                fetch_urls_df,
                MAX_URLS_PER_POLYGON,
            )
        ):
            logger.info(
                "Loaded %s candidate URLs and %s fetch URLs",
                len(candidate_urls_df),
                len(fetch_urls_df),
            )
            return candidate_urls_df, fetch_urls_df, False

        logger.info("Rebuilding candidate URL artifacts because metadata is incomplete")

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


def enrich_sentence_metadata(
    sentence_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> pd.DataFrame:
    if sentence_df.empty:
        return sentence_df

    return attach_polygon_metadata(sentence_df, pilot_gdf)


def build_pilot_sentence_candidates(
    page_text_with_quality_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> pd.DataFrame:
    sentence_df = build_sentence_candidate_dataframe(page_text_with_quality_df)
    sentence_df = enrich_sentence_metadata(sentence_df, pilot_gdf)
    sentence_df = deduplicate_near_duplicate_sentence_candidates(sentence_df)
    return select_complete_sentence_candidates(
        sentence_df,
        sentences_per_polygon=MAX_SENTENCES_PER_POLYGON,
        sentences_per_url=MAX_SENTENCES_PER_URL,
        target_polygon_count=TARGET_COMPLETE_POLYGON_COUNT,
    )


def load_or_build_sentence_candidates(
    path: Path,
    page_text_with_quality_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
    logger: logging.Logger,
    reset: bool,
) -> pd.DataFrame:
    if path.exists() and not reset:
        sentence_df = pd.read_parquet(path)
        if sentence_artifact_respects_sampling_limits(
            sentence_df,
            sentences_per_polygon=MAX_SENTENCES_PER_POLYGON,
            sentences_per_url=MAX_SENTENCES_PER_URL,
            target_polygon_count=TARGET_COMPLETE_POLYGON_COUNT,
        ):
            logger.info("Loaded %s rows for sentence candidates from %s", len(sentence_df), path)
            return sentence_df

        logger.info("Rebuilding sentence candidates because sampling limits changed")

    sentence_df = build_pilot_sentence_candidates(page_text_with_quality_df, pilot_gdf)
    sentence_df.to_parquet(path, index=False)
    logger.info("Saved %s rows for sentence candidates to %s", len(sentence_df), path)

    return sentence_df


def sentence_quota_is_satisfied(
    page_text_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> bool:
    if page_text_df.empty:
        return False

    page_text_with_quality_df = add_quality_metadata(page_text_df)
    sentence_df = build_pilot_sentence_candidates(page_text_with_quality_df, pilot_gdf)
    return sentence_artifact_respects_sampling_limits(
        sentence_df,
        sentences_per_polygon=MAX_SENTENCES_PER_POLYGON,
        sentences_per_url=MAX_SENTENCES_PER_URL,
        target_polygon_count=TARGET_COMPLETE_POLYGON_COUNT,
    )


def save_complete_sentence_polygons(
    sentence_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    complete_pilot_gdf = filter_to_sentence_polygons(pilot_gdf, sentence_df)
    save_geodataframe(complete_pilot_gdf, COMPLETE_POLYGONS_PATH)
    logger.info(
        "Saved %s complete sentence polygons to %s",
        len(complete_pilot_gdf),
        COMPLETE_POLYGONS_PATH,
    )

    return complete_pilot_gdf


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
        "max_queries_per_polygon=%s, max_urls_per_polygon=%s, "
        "target_complete_polygons=%s, max_sentences_per_polygon=%s, "
        "max_sentences_per_url=%s, fetch_timeout_seconds=%s",
        SAMPLE_SIZE,
        RESULTS_PER_QUERY,
        MAX_QUERIES_PER_POLYGON,
        MAX_URLS_PER_POLYGON,
        TARGET_COMPLETE_POLYGON_COUNT,
        MAX_SENTENCES_PER_POLYGON,
        MAX_SENTENCES_PER_URL,
        FETCH_TIMEOUT_SECONDS,
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

    page_text_reset = RESET_OUTPUTS
    page_text_df, page_text_changed = fetch_candidate_pages(
        fetch_urls_df,
        logger,
        output_path=PAGE_TEXT_PATH,
        reset=page_text_reset,
        stop_when=lambda dataframe: sentence_quota_is_satisfied(dataframe, pilot_gdf),
        fetch_timeout_seconds=FETCH_TIMEOUT_SECONDS,
        fetch_delay_seconds=FETCH_DELAY_SECONDS,
    )
    logger.info("Saved %s fetched page rows to %s", len(page_text_df), PAGE_TEXT_PATH)

    text_metadata_reset = page_text_reset or page_text_changed
    page_text_with_quality_df = load_or_build_page_text_quality(
        page_text_df,
        logger,
        reset=text_metadata_reset,
    )

    sentence_df = load_or_build_sentence_candidates(
        path=SENTENCE_CANDIDATES_PATH,
        page_text_with_quality_df=page_text_with_quality_df,
        pilot_gdf=pilot_gdf,
        logger=logger,
        reset=text_metadata_reset,
    )
    complete_pilot_gdf = save_complete_sentence_polygons(
        sentence_df,
        pilot_gdf,
        logger,
    )

    analysis = summarize_sentence_pilot(
        polygons_df=complete_pilot_gdf,
        search_results_df=filter_to_sentence_polygons(search_results_df, sentence_df),
        candidate_urls_df=filter_to_sentence_polygons(candidate_urls_df, sentence_df),
        page_text_df=filter_to_sentence_polygons(page_text_with_quality_df, sentence_df),
        sentence_df=sentence_df,
    )
    analysis.update(
        {
            "searched_polygon_count": int(len(pilot_gdf)),
            "target_complete_polygon_count": TARGET_COMPLETE_POLYGON_COUNT,
            "sentences_per_polygon_target": MAX_SENTENCES_PER_POLYGON,
            "sentences_per_url_target": MAX_SENTENCES_PER_URL,
            "sentence_deduplication_method": "minhash",
            "sentence_deduplication_threshold": MINHASH_DUPLICATE_THRESHOLD,
            "sentence_filter_profile": SENTENCE_FILTER_PROFILE,
            "sentence_filter_rules": list(SENTENCE_FILTER_RULES),
        }
    )
    write_analysis(analysis, logger)
    logger.info("Worldwide sentence pilot finished")


if __name__ == "__main__":
    main()

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from georeset_osm_web_evidence.evidence.page_text import fetch_candidate_pages
from georeset_osm_web_evidence.evidence.english_sentences import (
    build_english_sentence_candidates,
)
from georeset_osm_web_evidence.evidence.sentence_candidates import (
    MINHASH_DUPLICATE_THRESHOLD,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    attach_polygon_metadata,
    build_candidate_urls,
    build_search_rows_for_query,
    filter_to_sentence_polygons,
    summarize_sentence_pilot,
)
from georeset_osm_web_evidence.labeling.requests import (
    build_sentence_candidate_prompt_rows,
    write_labeling_prompt_jsonl,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import (
    build_contextual_english_search_queries,
)
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.text.sentences import (
    SENTENCE_FILTER_PROFILE,
    SENTENCE_FILTER_RULES,
)
from georeset_osm_web_evidence.web.quality import add_quality_metadata


ENGLISH_ONLY_OUTPUT_DIR = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only"
)
OUTPUT_DIR = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_context_queries_v1"
)

INPUT_POLYGONS_PATH = ENGLISH_ONLY_OUTPUT_DIR / "complete_sentence_polygons.parquet"
PILOT_POLYGONS_PATH = OUTPUT_DIR / "pilot_polygons.parquet"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "search_results.parquet"
SEARCH_ATTEMPTS_PATH = OUTPUT_DIR / "search_attempts.parquet"
CANDIDATE_URLS_PATH = OUTPUT_DIR / "candidate_urls.parquet"
PAGE_TEXT_PATH = OUTPUT_DIR / "page_text.parquet"
PAGE_TEXT_WITH_QUALITY_PATH = OUTPUT_DIR / "page_text_with_quality.parquet"
SENTENCE_CANDIDATES_PATH = OUTPUT_DIR / "sentence_candidates.parquet"
COMPLETE_POLYGONS_PATH = OUTPUT_DIR / "complete_sentence_polygons.parquet"
LLM_REQUESTS_PARQUET_PATH = OUTPUT_DIR / "llm_labeling_requests.parquet"
LLM_REQUESTS_JSONL_PATH = OUTPUT_DIR / "llm_labeling_requests.jsonl"
ANALYSIS_PATH = OUTPUT_DIR / "analysis.json"
LOG_PATH = OUTPUT_DIR / "run.log"

TARGET_POLYGON_COUNT = int(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_TARGET_POLYGONS", "10")
)
SENTENCES_PER_POLYGON = int(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_SENTENCES_PER_POLYGON", "10")
)
SENTENCES_PER_URL = int(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_SENTENCES_PER_URL", "1")
)
MAX_QUERIES_PER_POLYGON = int(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_MAX_QUERIES_PER_POLYGON", "4")
)
MAX_URLS_PER_POLYGON = int(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_MAX_URLS_PER_POLYGON", "30")
)
RESULTS_PER_QUERY = int(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_RESULTS_PER_QUERY", "20")
)
REQUEST_DELAY_SECONDS = float(
    os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_REQUEST_DELAY_SECONDS", "1.2")
)
RESET_OUTPUTS = os.environ.get("CONTEXT_QUERY_SENTENCE_PILOT_RESET_OUTPUTS", "0") == "1"


def configure_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("english_context_query_sentence_pilot")
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


def build_context_query_search_artifacts(
    pilot_gdf: pd.DataFrame,
    search_func: Callable[[str, int], list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    results_per_query: int = RESULTS_PER_QUERY,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    attempt_rows = []

    for polygon_index, polygon_row in enumerate(pilot_gdf.itertuples(), start=1):
        queries = build_contextual_english_search_queries(
            osm_tags=polygon_row.osm_tags,
            country=polygon_row.country,
            world_region=polygon_row.world_region,
            source_extract_id=polygon_row.source_extract_id,
            polygon_category=polygon_row.polygon_category,
            max_queries=MAX_QUERIES_PER_POLYGON,
        )

        if logger is not None:
            logger.info(
                "Searching polygon %s/%s: %s (%s context queries)",
                polygon_index,
                len(pilot_gdf),
                polygon_row.polygon_name,
                len(queries),
            )

        for query in queries:
            search_error = None
            try:
                results = search_func(
                    query,
                    count=results_per_query,
                    country="US",
                    search_lang="en",
                )
            except Exception as error:
                results = []
                search_error = str(error)
                if logger is not None:
                    logger.warning("Search failed for %s: %s", query, search_error)

            result_rows, attempt_row = build_search_rows_for_query(
                polygon_row=polygon_row,
                query_language="en",
                query=query,
                results=results,
                search_error=search_error,
            )
            rows.extend(result_rows)
            attempt_rows.append(attempt_row)
            sleep_func(request_delay_seconds)

    return pd.DataFrame(rows), pd.DataFrame(attempt_rows)


def run_context_pilot_labeling_request_build(
    input_path: str | Path = SENTENCE_CANDIDATES_PATH,
    parquet_output_path: str | Path = LLM_REQUESTS_PARQUET_PATH,
    jsonl_output_path: str | Path = LLM_REQUESTS_JSONL_PATH,
) -> pd.DataFrame:
    input_path = Path(input_path)
    parquet_output_path = Path(parquet_output_path)
    jsonl_output_path = Path(jsonl_output_path)

    sentence_df = pd.read_parquet(input_path)
    prompt_df = build_sentence_candidate_prompt_rows(sentence_df)

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_df.to_parquet(parquet_output_path, index=False)
    write_labeling_prompt_jsonl(prompt_df, jsonl_output_path)

    return prompt_df


def english_sentence_quota_is_satisfied(
    page_text_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> bool:
    if page_text_df.empty:
        return False

    page_text_with_quality_df = add_quality_metadata(page_text_df)
    sentence_df = build_english_sentence_candidates(
        page_text_with_quality_df,
        pilot_gdf,
        target_polygon_count=TARGET_POLYGON_COUNT,
        sentences_per_polygon=SENTENCES_PER_POLYGON,
        sentences_per_url=SENTENCES_PER_URL,
    )

    return len(sentence_df) == TARGET_POLYGON_COUNT * SENTENCES_PER_POLYGON


def write_analysis(analysis: dict, logger: logging.Logger) -> None:
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    logger.info("Analysis: %s", json.dumps(analysis, sort_keys=True))


def reset_outputs() -> None:
    for path in [
        PILOT_POLYGONS_PATH,
        SEARCH_RESULTS_PATH,
        SEARCH_ATTEMPTS_PATH,
        CANDIDATE_URLS_PATH,
        PAGE_TEXT_PATH,
        PAGE_TEXT_WITH_QUALITY_PATH,
        SENTENCE_CANDIDATES_PATH,
        COMPLETE_POLYGONS_PATH,
        LLM_REQUESTS_PARQUET_PATH,
        LLM_REQUESTS_JSONL_PATH,
        ANALYSIS_PATH,
    ]:
        path.unlink(missing_ok=True)


def main() -> None:
    logger = configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESET_OUTPUTS:
        reset_outputs()

    logger.info("Starting English context-query sentence pilot")
    pilot_gdf = load_geodataframe(INPUT_POLYGONS_PATH)
    save_geodataframe(pilot_gdf, PILOT_POLYGONS_PATH)
    logger.info("Pilot polygons: %s", len(pilot_gdf))

    if SEARCH_RESULTS_PATH.exists() and SEARCH_ATTEMPTS_PATH.exists() and not RESET_OUTPUTS:
        search_results_df = pd.read_parquet(SEARCH_RESULTS_PATH)
        search_attempts_df = pd.read_parquet(SEARCH_ATTEMPTS_PATH)
    else:
        search_results_df, search_attempts_df = build_context_query_search_artifacts(
            pilot_gdf,
            logger=logger,
        )
        search_results_df.to_parquet(SEARCH_RESULTS_PATH, index=False)
        search_attempts_df.to_parquet(SEARCH_ATTEMPTS_PATH, index=False)
    logger.info("Search results: %s", len(search_results_df))
    logger.info("Search attempts: %s", len(search_attempts_df))

    if CANDIDATE_URLS_PATH.exists() and not RESET_OUTPUTS:
        candidate_urls_df = pd.read_parquet(CANDIDATE_URLS_PATH)
    else:
        candidate_urls_df = build_candidate_urls(
            search_results_df,
            max_urls_per_polygon=MAX_URLS_PER_POLYGON,
        )
        candidate_urls_df["query_language"] = "en"
        candidate_urls_df = attach_polygon_metadata(candidate_urls_df, pilot_gdf)
        candidate_urls_df.to_parquet(CANDIDATE_URLS_PATH, index=False)
    logger.info("Candidate URLs: %s", len(candidate_urls_df))

    page_text_df, page_text_changed = fetch_candidate_pages(
        candidate_urls_df,
        logger,
        output_path=PAGE_TEXT_PATH,
        reset=False,
        stop_when=lambda dataframe: english_sentence_quota_is_satisfied(
            dataframe,
            pilot_gdf,
        ),
    )
    logger.info("Page text rows: %s", len(page_text_df))

    if PAGE_TEXT_WITH_QUALITY_PATH.exists() and not page_text_changed and not RESET_OUTPUTS:
        page_text_with_quality_df = pd.read_parquet(PAGE_TEXT_WITH_QUALITY_PATH)
    else:
        page_text_with_quality_df = add_quality_metadata(page_text_df)
        page_text_with_quality_df.to_parquet(PAGE_TEXT_WITH_QUALITY_PATH, index=False)
    logger.info("Quality rows: %s", len(page_text_with_quality_df))

    sentence_df = build_english_sentence_candidates(
        page_text_with_quality_df,
        pilot_gdf,
        target_polygon_count=TARGET_POLYGON_COUNT,
        sentences_per_polygon=SENTENCES_PER_POLYGON,
        sentences_per_url=SENTENCES_PER_URL,
    )
    sentence_df.to_parquet(SENTENCE_CANDIDATES_PATH, index=False)
    logger.info("Sentence candidates: %s", len(sentence_df))

    complete_pilot_gdf = filter_to_sentence_polygons(pilot_gdf, sentence_df)
    save_geodataframe(complete_pilot_gdf, COMPLETE_POLYGONS_PATH)
    logger.info("Complete polygons: %s", len(complete_pilot_gdf))

    prompt_df = run_context_pilot_labeling_request_build()
    logger.info("LLM prompt rows: %s", len(prompt_df))

    analysis = summarize_sentence_pilot(
        polygons_df=complete_pilot_gdf,
        search_results_df=filter_to_sentence_polygons(search_results_df, sentence_df),
        candidate_urls_df=filter_to_sentence_polygons(candidate_urls_df, sentence_df),
        page_text_df=filter_to_sentence_polygons(page_text_with_quality_df, sentence_df),
        sentence_df=sentence_df,
    )
    analysis.update(
        {
            "experiment_name": OUTPUT_DIR.name,
            "input_polygons_path": str(INPUT_POLYGONS_PATH),
            "query_language": "en",
            "query_strategy": "polygon name + location context + polygon category",
            "target_complete_polygon_count": TARGET_POLYGON_COUNT,
            "sentences_per_polygon_target": SENTENCES_PER_POLYGON,
            "sentences_per_url_target": SENTENCES_PER_URL,
            "max_queries_per_polygon": MAX_QUERIES_PER_POLYGON,
            "max_urls_per_polygon": MAX_URLS_PER_POLYGON,
            "sentence_deduplication_method": "minhash",
            "sentence_deduplication_threshold": MINHASH_DUPLICATE_THRESHOLD,
            "sentence_filter_profile": SENTENCE_FILTER_PROFILE,
            "sentence_filter_rules": list(SENTENCE_FILTER_RULES),
        }
    )
    write_analysis(analysis, logger)
    logger.info("English context-query sentence pilot finished")


if __name__ == "__main__":
    main()

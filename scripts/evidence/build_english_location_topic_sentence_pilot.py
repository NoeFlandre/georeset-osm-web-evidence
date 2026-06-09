import logging
import os
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.completion_candidates import (
    order_completion_candidates,
    polygon_keys,
)
from georeset_osm_web_evidence.evidence.english_sentences import (
    build_english_sentence_candidates,
    english_sentence_quota_is_satisfied,
)
from georeset_osm_web_evidence.evidence.final_url_artifacts import (
    select_exact_url_artifacts,
    validate_exact_sentence_url_counts,
)
from georeset_osm_web_evidence.evidence.location_topic_search import (
    build_location_topic_search_artifacts,
    empty_location_topic_search_attempts,
    empty_location_topic_search_results,
    search_location_topic_for_polygon,
)
from georeset_osm_web_evidence.evidence.page_text import fetch_candidate_pages
from georeset_osm_web_evidence.evidence.sentence_candidates import (
    MINHASH_DUPLICATE_THRESHOLD,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    attach_polygon_metadata,
    build_candidate_urls,
    filter_to_sentence_polygons,
    summarize_sentence_pilot,
)
from georeset_osm_web_evidence.labeling.requests import (
    build_and_write_labeling_prompt_artifacts,
    build_location_aware_sentence_candidate_prompt_rows,
)
from georeset_osm_web_evidence.pipeline.artifacts import (
    delete_artifacts,
    write_json_artifact,
)
from georeset_osm_web_evidence.pipeline.logging import configure_stage_logger
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.storage.dataframe import append_unique_rows
from georeset_osm_web_evidence.text.sentences import (
    SENTENCE_FILTER_PROFILE,
    SENTENCE_FILTER_RULES,
)
from georeset_osm_web_evidence.web.quality import add_quality_metadata


ENGLISH_ONLY_OUTPUT_DIR = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only"
)
SOURCE_POLYGONS_PATH = Path(
    "data/processed/samples/worldwide_training_polygon_sample.parquet"
)
OUTPUT_DIR = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_location_topic_queries_v1"
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
FINAL_SEARCH_RESULTS_PATH = OUTPUT_DIR / "final_search_results.parquet"
FINAL_CANDIDATE_URLS_PATH = OUTPUT_DIR / "final_candidate_urls.parquet"
FINAL_PAGE_TEXT_PATH = OUTPUT_DIR / "final_page_text.parquet"
FINAL_PAGE_TEXT_WITH_QUALITY_PATH = OUTPUT_DIR / "final_page_text_with_quality.parquet"
ANALYSIS_PATH = OUTPUT_DIR / "analysis.json"
FINAL_ANALYSIS_PATH = OUTPUT_DIR / "final_analysis.json"
COMPLETION_REJECTED_POLYGONS_PATH = OUTPUT_DIR / "completion_rejected_polygons.parquet"
LOG_PATH = OUTPUT_DIR / "run.log"

TARGET_POLYGON_COUNT = int(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_TARGET_POLYGONS", "10")
)
SENTENCES_PER_POLYGON = int(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_SENTENCES_PER_POLYGON", "10")
)
SENTENCES_PER_URL = int(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_SENTENCES_PER_URL", "1")
)
MAX_QUERIES_PER_POLYGON = int(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_MAX_QUERIES_PER_POLYGON", "4")
)
MAX_URLS_PER_POLYGON = int(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_MAX_URLS_PER_POLYGON", "30")
)
RESULTS_PER_QUERY = int(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_RESULTS_PER_QUERY", "20")
)
REQUEST_DELAY_SECONDS = float(
    os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_REQUEST_DELAY_SECONDS", "1.2")
)
RESET_OUTPUTS = os.environ.get("LOCATION_TOPIC_SENTENCE_PILOT_RESET_OUTPUTS", "0") == "1"


def configure_logging() -> logging.Logger:
    return configure_stage_logger("english_location_topic_sentence_pilot", LOG_PATH)


def run_location_topic_labeling_request_build(
    input_path: str | Path = SENTENCE_CANDIDATES_PATH,
    parquet_output_path: str | Path = LLM_REQUESTS_PARQUET_PATH,
    jsonl_output_path: str | Path = LLM_REQUESTS_JSONL_PATH,
) -> pd.DataFrame:
    return build_and_write_labeling_prompt_artifacts(
        input_path=input_path,
        parquet_output_path=parquet_output_path,
        jsonl_output_path=jsonl_output_path,
        prompt_builder=build_location_aware_sentence_candidate_prompt_rows,
    )


def _metadata_ready(source_df: pd.DataFrame) -> bool:
    return all(
        column in source_df.columns
        for column in ["polygon_name", "polygon_category", "query_local_language"]
    )


def load_completion_source_polygons() -> pd.DataFrame:
    source_gdf = load_geodataframe(SOURCE_POLYGONS_PATH)
    if _metadata_ready(source_gdf):
        return source_gdf

    from georeset_osm_web_evidence.evidence.worldwide_pilot import add_pilot_metadata

    return add_pilot_metadata(source_gdf)


def _candidate_temp_path(polygon_row) -> Path:
    return OUTPUT_DIR / f"_tmp_page_text_{polygon_row.osm_type}_{polygon_row.osm_id}.parquet"


def build_complete_sentences_for_single_polygon(
    candidate_urls_df: pd.DataFrame,
    pilot_row_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_path = _candidate_temp_path(pilot_row_df.iloc[0])
    temp_path.unlink(missing_ok=True)

    page_text_df, _ = fetch_candidate_pages(
        candidate_urls_df,
        logger,
        output_path=temp_path,
        reset=True,
    )
    page_text_with_quality_df = add_quality_metadata(page_text_df)
    sentence_df = build_english_sentence_candidates(
        page_text_with_quality_df,
        pilot_row_df,
        target_polygon_count=1,
        sentences_per_polygon=SENTENCES_PER_POLYGON,
        sentences_per_url=SENTENCES_PER_URL,
    )
    temp_path.unlink(missing_ok=True)

    return sentence_df, page_text_df, page_text_with_quality_df


def complete_location_topic_pilot(
    target_polygon_count: int = TARGET_POLYGON_COUNT,
    urls_per_polygon: int = SENTENCES_PER_POLYGON,
    logger: logging.Logger | None = None,
) -> dict:
    logger = logger or configure_logging()
    source_df = load_completion_source_polygons()
    current_sentence_df = pd.read_parquet(SENTENCE_CANDIDATES_PATH)
    current_complete_df = load_geodataframe(COMPLETE_POLYGONS_PATH)
    current_candidate_urls_df = pd.read_parquet(
        FINAL_CANDIDATE_URLS_PATH
        if FINAL_CANDIDATE_URLS_PATH.exists()
        else CANDIDATE_URLS_PATH
    )
    current_page_text_df = pd.read_parquet(
        FINAL_PAGE_TEXT_PATH if FINAL_PAGE_TEXT_PATH.exists() else PAGE_TEXT_PATH
    )
    current_page_text_with_quality_df = pd.read_parquet(
        FINAL_PAGE_TEXT_WITH_QUALITY_PATH
        if FINAL_PAGE_TEXT_WITH_QUALITY_PATH.exists()
        else PAGE_TEXT_WITH_QUALITY_PATH
    )
    search_results_df = (
        pd.read_parquet(SEARCH_RESULTS_PATH)
        if SEARCH_RESULTS_PATH.exists()
        else empty_location_topic_search_results()
    )
    search_attempts_df = (
        pd.read_parquet(SEARCH_ATTEMPTS_PATH)
        if SEARCH_ATTEMPTS_PATH.exists()
        else empty_location_topic_search_attempts()
    )

    final_candidate_urls_df, final_page_text_df = select_exact_url_artifacts(
        sentence_df=current_sentence_df,
        candidate_urls_df=current_candidate_urls_df,
        page_text_df=current_page_text_df,
        urls_per_polygon=urls_per_polygon,
    )
    _, final_page_text_with_quality_df = select_exact_url_artifacts(
        sentence_df=current_sentence_df,
        candidate_urls_df=current_candidate_urls_df,
        page_text_df=current_page_text_with_quality_df,
        urls_per_polygon=urls_per_polygon,
    )

    rejected_rows = []
    attempted_df = search_attempts_df[["osm_type", "osm_id"]].drop_duplicates()
    complete_df = current_complete_df.copy()
    sentence_df = current_sentence_df.copy()
    candidate_urls_df = final_candidate_urls_df.copy()
    page_text_df = final_page_text_df.copy()
    page_text_with_quality_df = final_page_text_with_quality_df.copy()

    while len(polygon_keys(complete_df)) < target_polygon_count:
        candidates_df = order_completion_candidates(
            source_df=source_df,
            complete_df=complete_df,
            attempted_df=attempted_df,
        )
        if candidates_df.empty:
            raise ValueError("No completion candidates left to try")

        polygon_row = next(candidates_df.itertuples())
        logger.info(
            "Trying completion polygon %s/%s: %s (%s, %s)",
            len(polygon_keys(complete_df)) + 1,
            target_polygon_count,
            polygon_row.polygon_name,
            polygon_row.world_region,
            polygon_row.area_size_bin,
        )

        new_results_df, new_attempts_df = search_location_topic_for_polygon(
            polygon_row,
            max_queries_per_polygon=MAX_QUERIES_PER_POLYGON,
            results_per_query=RESULTS_PER_QUERY,
            request_delay_seconds=REQUEST_DELAY_SECONDS,
            logger=logger,
        )
        search_results_df = append_unique_rows(
            search_results_df,
            new_results_df,
            subset=["osm_type", "osm_id", "query", "url"],
        )
        search_attempts_df = append_unique_rows(
            search_attempts_df,
            new_attempts_df,
            subset=["osm_type", "osm_id", "query"],
        )
        attempted_df = search_attempts_df[["osm_type", "osm_id"]].drop_duplicates()
        search_results_df.to_parquet(SEARCH_RESULTS_PATH, index=False)
        search_attempts_df.to_parquet(SEARCH_ATTEMPTS_PATH, index=False)

        polygon_candidate_urls_df = build_candidate_urls(
            new_results_df,
            max_urls_per_polygon=MAX_URLS_PER_POLYGON,
        )
        if polygon_candidate_urls_df.empty or len(polygon_candidate_urls_df) < urls_per_polygon:
            rejected_rows.append(
                {
                    "osm_type": polygon_row.osm_type,
                    "osm_id": polygon_row.osm_id,
                    "polygon_name": polygon_row.polygon_name,
                    "reason": "fewer_than_10_candidate_urls",
                    "candidate_url_count": len(polygon_candidate_urls_df),
                }
            )
            pd.DataFrame(rejected_rows).to_parquet(
                COMPLETION_REJECTED_POLYGONS_PATH,
                index=False,
            )
            continue

        pilot_row_df = pd.DataFrame([polygon_row._asdict()])
        polygon_candidate_urls_df["query_language"] = "en"
        polygon_candidate_urls_df = attach_polygon_metadata(
            polygon_candidate_urls_df,
            pilot_row_df,
        )
        polygon_candidate_urls_df = polygon_candidate_urls_df.head(MAX_URLS_PER_POLYGON)

        polygon_sentence_df, polygon_page_text_df, polygon_page_text_with_quality_df = (
            build_complete_sentences_for_single_polygon(
                polygon_candidate_urls_df,
                pilot_row_df,
                logger,
            )
        )
        if len(polygon_sentence_df) != urls_per_polygon:
            rejected_rows.append(
                {
                    "osm_type": polygon_row.osm_type,
                    "osm_id": polygon_row.osm_id,
                    "polygon_name": polygon_row.polygon_name,
                    "reason": "fewer_than_10_sentence_urls",
                    "candidate_url_count": len(polygon_candidate_urls_df),
                    "sentence_count": len(polygon_sentence_df),
                }
            )
            pd.DataFrame(rejected_rows).to_parquet(
                COMPLETION_REJECTED_POLYGONS_PATH,
                index=False,
            )
            continue

        polygon_final_candidate_urls_df, polygon_final_page_text_df = (
            select_exact_url_artifacts(
                sentence_df=polygon_sentence_df,
                candidate_urls_df=polygon_candidate_urls_df,
                page_text_df=polygon_page_text_df,
                urls_per_polygon=urls_per_polygon,
            )
        )
        _, polygon_final_page_text_with_quality_df = select_exact_url_artifacts(
            sentence_df=polygon_sentence_df,
            candidate_urls_df=polygon_candidate_urls_df,
            page_text_df=polygon_page_text_with_quality_df,
            urls_per_polygon=urls_per_polygon,
        )

        complete_df = pd.concat([complete_df, pilot_row_df], ignore_index=True)
        sentence_df = pd.concat([sentence_df, polygon_sentence_df], ignore_index=True)
        candidate_urls_df = pd.concat(
            [candidate_urls_df, polygon_final_candidate_urls_df],
            ignore_index=True,
        )
        page_text_df = pd.concat(
            [page_text_df, polygon_final_page_text_df],
            ignore_index=True,
        )
        page_text_with_quality_df = pd.concat(
            [page_text_with_quality_df, polygon_final_page_text_with_quality_df],
            ignore_index=True,
        )

        save_geodataframe(complete_df, COMPLETE_POLYGONS_PATH)
        sentence_df.to_parquet(SENTENCE_CANDIDATES_PATH, index=False)
        candidate_urls_df.to_parquet(FINAL_CANDIDATE_URLS_PATH, index=False)
        page_text_df.to_parquet(FINAL_PAGE_TEXT_PATH, index=False)
        page_text_with_quality_df.to_parquet(
            FINAL_PAGE_TEXT_WITH_QUALITY_PATH,
            index=False,
        )
        logger.info("Accepted completion polygon: %s", polygon_row.polygon_name)

    validate_exact_sentence_url_counts(
        sentence_df,
        urls_per_polygon=urls_per_polygon,
    )
    prompt_df = run_location_topic_labeling_request_build()
    final_search_results_df = search_results_df.merge(
        candidate_urls_df[["osm_type", "osm_id", "url"]].drop_duplicates(),
        on=["osm_type", "osm_id", "url"],
        how="inner",
    )
    final_search_results_df.to_parquet(FINAL_SEARCH_RESULTS_PATH, index=False)
    candidate_urls_df.to_parquet(FINAL_CANDIDATE_URLS_PATH, index=False)
    page_text_df.to_parquet(FINAL_PAGE_TEXT_PATH, index=False)
    page_text_with_quality_df.to_parquet(
        FINAL_PAGE_TEXT_WITH_QUALITY_PATH,
        index=False,
    )

    final_analysis = summarize_sentence_pilot(
        polygons_df=complete_df,
        search_results_df=final_search_results_df,
        candidate_urls_df=candidate_urls_df,
        page_text_df=page_text_with_quality_df,
        sentence_df=sentence_df,
    )
    final_analysis.update(
        {
            "experiment_name": OUTPUT_DIR.name,
            "url_selection": "URLs used by final selected sentences",
            "urls_per_polygon": urls_per_polygon,
            "sentences_per_polygon": urls_per_polygon,
            "sentences_per_url": SENTENCES_PER_URL,
            "llm_prompt_rows": int(len(prompt_df)),
        }
    )
    write_json_artifact(FINAL_ANALYSIS_PATH, final_analysis)

    return final_analysis


def finalize_existing_location_topic_outputs(
    urls_per_polygon: int = SENTENCES_PER_POLYGON,
) -> dict:
    sentence_df = pd.read_parquet(SENTENCE_CANDIDATES_PATH)
    candidate_urls_df = pd.read_parquet(CANDIDATE_URLS_PATH)
    page_text_df = pd.read_parquet(PAGE_TEXT_PATH)
    page_text_with_quality_df = pd.read_parquet(PAGE_TEXT_WITH_QUALITY_PATH)
    search_results_df = pd.read_parquet(SEARCH_RESULTS_PATH)
    complete_pilot_gdf = load_geodataframe(COMPLETE_POLYGONS_PATH)

    final_candidate_urls_df, final_page_text_df = select_exact_url_artifacts(
        sentence_df=sentence_df,
        candidate_urls_df=candidate_urls_df,
        page_text_df=page_text_df,
        urls_per_polygon=urls_per_polygon,
    )
    _, final_page_text_with_quality_df = select_exact_url_artifacts(
        sentence_df=sentence_df,
        candidate_urls_df=candidate_urls_df,
        page_text_df=page_text_with_quality_df,
        urls_per_polygon=urls_per_polygon,
    )
    final_search_results_df = search_results_df.merge(
        final_candidate_urls_df[["osm_type", "osm_id", "url"]].drop_duplicates(),
        on=["osm_type", "osm_id", "url"],
        how="inner",
    )

    final_search_results_df.to_parquet(FINAL_SEARCH_RESULTS_PATH, index=False)
    final_candidate_urls_df.to_parquet(FINAL_CANDIDATE_URLS_PATH, index=False)
    final_page_text_df.to_parquet(FINAL_PAGE_TEXT_PATH, index=False)
    final_page_text_with_quality_df.to_parquet(
        FINAL_PAGE_TEXT_WITH_QUALITY_PATH,
        index=False,
    )

    final_analysis = summarize_sentence_pilot(
        polygons_df=complete_pilot_gdf,
        search_results_df=final_search_results_df,
        candidate_urls_df=final_candidate_urls_df,
        page_text_df=final_page_text_with_quality_df,
        sentence_df=sentence_df,
    )
    final_analysis.update(
        {
            "experiment_name": OUTPUT_DIR.name,
            "url_selection": "URLs used by final selected sentences",
            "urls_per_polygon": urls_per_polygon,
            "sentences_per_polygon": urls_per_polygon,
            "sentences_per_url": SENTENCES_PER_URL,
        }
    )
    write_json_artifact(FINAL_ANALYSIS_PATH, final_analysis)

    return final_analysis


def reset_outputs() -> None:
    delete_artifacts(
        [
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
            FINAL_SEARCH_RESULTS_PATH,
            FINAL_CANDIDATE_URLS_PATH,
            FINAL_PAGE_TEXT_PATH,
            FINAL_PAGE_TEXT_WITH_QUALITY_PATH,
            ANALYSIS_PATH,
            FINAL_ANALYSIS_PATH,
        ]
    )


def main() -> None:
    logger = configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESET_OUTPUTS:
        reset_outputs()

    logger.info("Starting English location-topic sentence pilot")
    pilot_gdf = load_geodataframe(INPUT_POLYGONS_PATH)
    save_geodataframe(pilot_gdf, PILOT_POLYGONS_PATH)
    logger.info("Pilot polygons: %s", len(pilot_gdf))

    if SEARCH_RESULTS_PATH.exists() and SEARCH_ATTEMPTS_PATH.exists() and not RESET_OUTPUTS:
        search_results_df = pd.read_parquet(SEARCH_RESULTS_PATH)
        search_attempts_df = pd.read_parquet(SEARCH_ATTEMPTS_PATH)
    else:
        search_results_df, search_attempts_df = build_location_topic_search_artifacts(
            pilot_gdf,
            max_queries_per_polygon=MAX_QUERIES_PER_POLYGON,
            results_per_query=RESULTS_PER_QUERY,
            request_delay_seconds=REQUEST_DELAY_SECONDS,
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
            target_polygon_count=TARGET_POLYGON_COUNT,
            sentences_per_polygon=SENTENCES_PER_POLYGON,
            sentences_per_url=SENTENCES_PER_URL,
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

    prompt_df = run_location_topic_labeling_request_build()
    logger.info("LLM prompt rows: %s", len(prompt_df))
    final_analysis = finalize_existing_location_topic_outputs(
        urls_per_polygon=SENTENCES_PER_POLYGON,
    )
    write_json_artifact(
        FINAL_ANALYSIS_PATH,
        final_analysis,
        logger=logger,
        log_label="Final exact URL analysis",
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
            "experiment_name": OUTPUT_DIR.name,
            "input_polygons_path": str(INPUT_POLYGONS_PATH),
            "query_language": "en",
            "query_strategy": (
                '"{polygon_name}" "{country_or_region}" "{topic_term}"'
            ),
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
    write_json_artifact(ANALYSIS_PATH, analysis, logger=logger, log_label="Analysis")
    logger.info("English location-topic sentence pilot finished")


if __name__ == "__main__":
    main()

import json
import logging
import os
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.sentence_candidates import (
    MINHASH_DUPLICATE_THRESHOLD,
    build_sentence_candidate_dataframe,
    deduplicate_near_duplicate_sentence_candidates,
    filter_english_sentence_candidates,
    select_complete_sentence_candidates,
    sentence_artifact_respects_sampling_limits,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    attach_polygon_metadata,
    build_candidate_urls,
    filter_to_sentence_polygons,
    summarize_sentence_pilot,
)
from georeset_osm_web_evidence.evidence.page_text import (
    PAGE_TEXT_COLUMNS,
    fetch_candidate_pages,
)
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.text.sentences import (
    SENTENCE_FILTER_PROFILE,
    SENTENCE_FILTER_RULES,
)
from georeset_osm_web_evidence.web.quality import add_quality_metadata


BASE_OUTPUT_DIR = Path("data/processed/pilots/worldwide_sentence_pilot_10")
OUTPUT_DIR = Path("data/processed/pilots/worldwide_sentence_pilot_10_english_only")

BASE_PILOT_POLYGONS_PATH = BASE_OUTPUT_DIR / "pilot_polygons.parquet"
BASE_SEARCH_RESULTS_PATH = BASE_OUTPUT_DIR / "search_results.parquet"
BASE_PAGE_TEXT_PATH = BASE_OUTPUT_DIR / "page_text.parquet"

CANDIDATE_URLS_PATH = OUTPUT_DIR / "candidate_urls.parquet"
PAGE_TEXT_PATH = OUTPUT_DIR / "page_text.parquet"
PAGE_TEXT_WITH_QUALITY_PATH = OUTPUT_DIR / "page_text_with_quality.parquet"
SENTENCE_CANDIDATES_PATH = OUTPUT_DIR / "sentence_candidates.parquet"
COMPLETE_POLYGONS_PATH = OUTPUT_DIR / "complete_sentence_polygons.parquet"
ANALYSIS_PATH = OUTPUT_DIR / "analysis.json"
LOG_PATH = OUTPUT_DIR / "run.log"

TARGET_POLYGON_COUNT = int(
    os.environ.get("ENGLISH_SENTENCE_PILOT_TARGET_POLYGONS", "10")
)
SENTENCES_PER_POLYGON = int(
    os.environ.get("ENGLISH_SENTENCE_PILOT_SENTENCES_PER_POLYGON", "10")
)
SENTENCES_PER_URL = int(os.environ.get("ENGLISH_SENTENCE_PILOT_SENTENCES_PER_URL", "1"))
MAX_URLS_PER_POLYGON = int(
    os.environ.get("ENGLISH_SENTENCE_PILOT_MAX_URLS_PER_POLYGON", "30")
)
RESET_OUTPUTS = os.environ.get("ENGLISH_SENTENCE_PILOT_RESET_OUTPUTS", "0") == "1"


def configure_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("english_sentence_pilot")
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


def filter_english_candidate_urls(candidate_urls_df: pd.DataFrame) -> pd.DataFrame:
    return candidate_urls_df[candidate_urls_df["query_language"].eq("en")].reset_index(
        drop=True
    )


def build_english_candidate_urls(
    search_results_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> pd.DataFrame:
    english_search_results_df = search_results_df[
        search_results_df["query_language"].eq("en")
    ]
    candidate_urls_df = build_candidate_urls(
        english_search_results_df,
        max_urls_per_polygon=MAX_URLS_PER_POLYGON,
    )
    candidate_urls_df["query_language"] = "en"

    return attach_polygon_metadata(candidate_urls_df, pilot_gdf)


def seed_page_text_from_base_cache(
    base_page_text_df: pd.DataFrame,
    candidate_urls_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_urls = set(candidate_urls_df["url"])
    seeded_df = base_page_text_df[base_page_text_df["source_url"].isin(candidate_urls)]
    seeded_df = seeded_df.copy()
    seeded_df["query_language"] = "en"

    for column in PAGE_TEXT_COLUMNS:
        if column not in seeded_df.columns:
            seeded_df[column] = pd.NA

    return seeded_df[PAGE_TEXT_COLUMNS].reset_index(drop=True)


def build_english_sentence_candidates(
    page_text_with_quality_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
    target_polygon_count: int = TARGET_POLYGON_COUNT,
    sentences_per_polygon: int = SENTENCES_PER_POLYGON,
    sentences_per_url: int = SENTENCES_PER_URL,
) -> pd.DataFrame:
    sentence_df = build_sentence_candidate_dataframe(page_text_with_quality_df)
    sentence_df = attach_polygon_metadata(sentence_df, pilot_gdf)
    sentence_df = filter_english_sentence_candidates(sentence_df)
    sentence_df = deduplicate_near_duplicate_sentence_candidates(sentence_df)

    return select_complete_sentence_candidates(
        sentence_df,
        sentences_per_polygon=sentences_per_polygon,
        sentences_per_url=sentences_per_url,
        target_polygon_count=target_polygon_count,
    )


def english_sentence_quota_is_satisfied(
    page_text_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> bool:
    if page_text_df.empty:
        return False

    page_text_with_quality_df = add_quality_metadata(page_text_df)
    sentence_df = build_english_sentence_candidates(page_text_with_quality_df, pilot_gdf)

    return sentence_artifact_respects_sampling_limits(
        sentence_df,
        sentences_per_polygon=SENTENCES_PER_POLYGON,
        sentences_per_url=SENTENCES_PER_URL,
        target_polygon_count=TARGET_POLYGON_COUNT,
    )


def write_analysis(analysis: dict, logger: logging.Logger) -> None:
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    logger.info("Analysis: %s", json.dumps(analysis, sort_keys=True))


def main() -> None:
    logger = configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESET_OUTPUTS:
        for path in [
            CANDIDATE_URLS_PATH,
            PAGE_TEXT_PATH,
            PAGE_TEXT_WITH_QUALITY_PATH,
            SENTENCE_CANDIDATES_PATH,
            COMPLETE_POLYGONS_PATH,
            ANALYSIS_PATH,
        ]:
            path.unlink(missing_ok=True)

    logger.info("Starting English-only sentence pilot")
    pilot_gdf = load_geodataframe(BASE_PILOT_POLYGONS_PATH)
    search_results_df = pd.read_parquet(BASE_SEARCH_RESULTS_PATH)

    if CANDIDATE_URLS_PATH.exists() and not RESET_OUTPUTS:
        candidate_urls_df = pd.read_parquet(CANDIDATE_URLS_PATH)
    else:
        candidate_urls_df = build_english_candidate_urls(search_results_df, pilot_gdf)
        candidate_urls_df.to_parquet(CANDIDATE_URLS_PATH, index=False)
    logger.info("English candidate URLs: %s", len(candidate_urls_df))

    if not PAGE_TEXT_PATH.exists() or RESET_OUTPUTS:
        base_page_text_df = pd.read_parquet(BASE_PAGE_TEXT_PATH)
        seeded_page_text_df = seed_page_text_from_base_cache(
            base_page_text_df,
            candidate_urls_df,
        )
        seeded_page_text_df.to_parquet(PAGE_TEXT_PATH, index=False)
        logger.info("Seeded %s English page rows", len(seeded_page_text_df))

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
    logger.info("English page rows: %s", len(page_text_df))

    if PAGE_TEXT_WITH_QUALITY_PATH.exists() and not page_text_changed and not RESET_OUTPUTS:
        page_text_with_quality_df = pd.read_parquet(PAGE_TEXT_WITH_QUALITY_PATH)
    else:
        page_text_with_quality_df = add_quality_metadata(page_text_df)
        page_text_with_quality_df.to_parquet(PAGE_TEXT_WITH_QUALITY_PATH, index=False)
    logger.info("English quality rows: %s", len(page_text_with_quality_df))

    sentence_df = build_english_sentence_candidates(page_text_with_quality_df, pilot_gdf)
    sentence_df.to_parquet(SENTENCE_CANDIDATES_PATH, index=False)
    logger.info("English sentence candidates: %s", len(sentence_df))

    complete_pilot_gdf = filter_to_sentence_polygons(pilot_gdf, sentence_df)
    save_geodataframe(complete_pilot_gdf, COMPLETE_POLYGONS_PATH)
    logger.info("English complete polygons: %s", len(complete_pilot_gdf))

    analysis = summarize_sentence_pilot(
        polygons_df=complete_pilot_gdf,
        search_results_df=filter_to_sentence_polygons(search_results_df, sentence_df),
        candidate_urls_df=filter_to_sentence_polygons(candidate_urls_df, sentence_df),
        page_text_df=filter_to_sentence_polygons(page_text_with_quality_df, sentence_df),
        sentence_df=sentence_df,
    )
    analysis.update(
        {
            "english_only_definition": (
                "query_language == 'en' and sentence passes the English heuristic"
            ),
            "target_complete_polygon_count": TARGET_POLYGON_COUNT,
            "sentences_per_polygon_target": SENTENCES_PER_POLYGON,
            "sentences_per_url_target": SENTENCES_PER_URL,
            "sentence_deduplication_method": "minhash",
            "sentence_deduplication_threshold": MINHASH_DUPLICATE_THRESHOLD,
            "sentence_filter_profile": SENTENCE_FILTER_PROFILE,
            "sentence_filter_rules": list(SENTENCE_FILTER_RULES),
        }
    )
    write_analysis(analysis, logger)
    logger.info("English-only sentence pilot finished")


if __name__ == "__main__":
    main()

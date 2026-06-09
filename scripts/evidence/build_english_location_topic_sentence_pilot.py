import json
import logging
import os
import time
import unicodedata
from pathlib import Path
from typing import Callable

import pandas as pd

from georeset_osm_web_evidence.evidence.page_text import fetch_candidate_pages
from georeset_osm_web_evidence.evidence.sentence_candidates import (
    sentence_artifact_respects_sampling_limits,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import (
    attach_polygon_metadata,
    build_candidate_urls,
    build_search_rows_for_query,
    filter_to_sentence_polygons,
    summarize_sentence_pilot,
)
from georeset_osm_web_evidence.labeling.requests import (
    build_location_aware_sentence_candidate_prompt_rows,
    write_labeling_prompt_jsonl,
)
from georeset_osm_web_evidence.search.providers import search_brave
from georeset_osm_web_evidence.search.queries import (
    build_location_topic_english_search_queries,
)
from georeset_osm_web_evidence.storage.local import load_geodataframe, save_geodataframe
from georeset_osm_web_evidence.web.quality import add_quality_metadata
from scripts.evidence.build_english_only_sentence_pilot import (
    MINHASH_DUPLICATE_THRESHOLD,
    SENTENCE_FILTER_PROFILE,
    SENTENCE_FILTER_RULES,
    build_english_sentence_candidates,
)


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("english_location_topic_sentence_pilot")
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


def build_location_topic_search_artifacts(
    pilot_gdf: pd.DataFrame,
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    results_per_query: int = RESULTS_PER_QUERY,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    attempt_rows = []

    for polygon_index, polygon_row in enumerate(pilot_gdf.itertuples(), start=1):
        queries = build_location_topic_english_search_queries(
            osm_tags=polygon_row.osm_tags,
            country=polygon_row.country,
            world_region=polygon_row.world_region,
            source_extract_id=polygon_row.source_extract_id,
            polygon_category=polygon_row.polygon_category,
            max_queries=MAX_QUERIES_PER_POLYGON,
        )

        if logger is not None:
            logger.info(
                "Searching polygon %s/%s: %s (%s location-topic queries)",
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


def run_location_topic_labeling_request_build(
    input_path: str | Path = SENTENCE_CANDIDATES_PATH,
    parquet_output_path: str | Path = LLM_REQUESTS_PARQUET_PATH,
    jsonl_output_path: str | Path = LLM_REQUESTS_JSONL_PATH,
) -> pd.DataFrame:
    input_path = Path(input_path)
    parquet_output_path = Path(parquet_output_path)
    jsonl_output_path = Path(jsonl_output_path)

    sentence_df = pd.read_parquet(input_path)
    prompt_df = build_location_aware_sentence_candidate_prompt_rows(sentence_df)

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_df.to_parquet(parquet_output_path, index=False)
    write_labeling_prompt_jsonl(prompt_df, jsonl_output_path)

    return prompt_df


def _sentence_url_counts(sentence_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sentence_df.groupby(["osm_type", "osm_id"], dropna=False)
        .agg(sentence_count=("sentence", "size"), url_count=("url", "nunique"))
        .reset_index()
    )


def validate_exact_sentence_url_counts(
    sentence_df: pd.DataFrame,
    urls_per_polygon: int,
) -> None:
    counts_df = _sentence_url_counts(sentence_df)
    invalid_counts_df = counts_df[
        (counts_df["sentence_count"] != urls_per_polygon)
        | (counts_df["url_count"] != urls_per_polygon)
    ]

    if not invalid_counts_df.empty:
        raise ValueError(
            "Selected sentence artifact does not have exactly "
            f"{urls_per_polygon} sentences and URLs per polygon: "
            f"{invalid_counts_df.to_dict('records')}"
        )


def select_exact_url_artifacts(
    sentence_df: pd.DataFrame,
    candidate_urls_df: pd.DataFrame,
    page_text_df: pd.DataFrame,
    urls_per_polygon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_exact_sentence_url_counts(sentence_df, urls_per_polygon=urls_per_polygon)

    sentence_urls_df = sentence_df[["osm_type", "osm_id", "url"]].drop_duplicates()
    selected_candidate_urls_df = sentence_urls_df.merge(
        candidate_urls_df,
        on=["osm_type", "osm_id", "url"],
        how="left",
    )
    page_text_keys_df = sentence_urls_df.rename(columns={"url": "source_url"})
    selected_page_text_df = page_text_keys_df.merge(
        page_text_df,
        on=["osm_type", "osm_id", "source_url"],
        how="left",
    )

    if selected_candidate_urls_df["best_rank"].isna().any():
        raise ValueError("Some selected sentence URLs are missing from candidate URLs")
    if selected_page_text_df["source_url"].isna().any():
        raise ValueError("Some selected sentence URLs are missing from page text")

    return selected_candidate_urls_df, selected_page_text_df


def _polygon_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["osm_type", "osm_id"])

    return df[["osm_type", "osm_id"]].drop_duplicates()


def _metadata_ready(source_df: pd.DataFrame) -> bool:
    return all(
        column in source_df.columns
        for column in ["polygon_name", "polygon_category", "query_local_language"]
    )


def _name_quality_score(name: object) -> int:
    if not isinstance(name, str):
        return -100

    normalized_name = " ".join(name.lower().split())
    generic_names = {
        "group of trees",
        "rice",
        "savannah",
        "forest",
        "wood",
        "woods",
        "wetland",
        "meadow",
    }
    if normalized_name in generic_names:
        return -50

    score = len(normalized_name)
    if " " in normalized_name:
        score += 20
    if any(character.isdigit() for character in normalized_name):
        score -= 10

    return score


def _has_osm_knowledge_graph_tag(osm_tags: object) -> int:
    if not isinstance(osm_tags, dict):
        return 0

    for key in ["wikipedia", "wikidata"]:
        value = osm_tags.get(key)
        if isinstance(value, str) and value.strip():
            return 1

    return 0


def _latin_name_score(name: object) -> int:
    if not isinstance(name, str) or not name.strip():
        return 0

    letters = [character for character in name if character.isalpha()]
    if not letters:
        return 0

    latin_letters = [
        character
        for character in letters
        if "LATIN" in unicodedata.name(character, "")
    ]

    return int(len(latin_letters) / len(letters) >= 0.7)


def _high_yield_place_name_score(name: object) -> int:
    if not isinstance(name, str):
        return 0

    normalized_name = name.lower()
    high_yield_terms = [
        "national park",
        "wildlife refuge",
        "nature reserve",
        "bird sanctuary",
        "conservation park",
        "state park",
        "provincial park",
        "natural reserve",
        "wildlife management area",
    ]

    return int(any(term in normalized_name for term in high_yield_terms))


def order_completion_candidates(
    source_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    attempted_df: pd.DataFrame,
) -> pd.DataFrame:
    complete_keys_df = _polygon_keys(complete_df)
    attempted_keys_df = _polygon_keys(attempted_df)
    excluded_keys_df = pd.concat(
        [complete_keys_df, attempted_keys_df],
        ignore_index=True,
    ).drop_duplicates()

    if excluded_keys_df.empty:
        remaining_df = source_df.copy()
    else:
        remaining_df = source_df.merge(
            excluded_keys_df.assign(_exclude=True),
            on=["osm_type", "osm_id"],
            how="left",
        )
        remaining_df = remaining_df[remaining_df["_exclude"].isna()].drop(
            columns=["_exclude"]
        )

    region_counts = complete_df["world_region"].value_counts().to_dict()
    area_bin_counts = complete_df["area_size_bin"].value_counts().to_dict()
    attempted_metadata_df = _polygon_keys(attempted_df).merge(
        source_df[["osm_type", "osm_id", "world_region", "area_size_bin"]],
        on=["osm_type", "osm_id"],
        how="left",
    )
    attempted_region_counts = (
        attempted_metadata_df["world_region"].value_counts().to_dict()
    )
    attempted_area_bin_counts = (
        attempted_metadata_df["area_size_bin"].value_counts().to_dict()
    )
    result = remaining_df.copy()
    result["_region_score"] = (
        result["world_region"].map(region_counts).fillna(0)
        + result["world_region"].map(attempted_region_counts).fillna(0) * 0.25
    )
    result["_area_bin_score"] = (
        result["area_size_bin"].map(area_bin_counts).fillna(0)
        + result["area_size_bin"].map(attempted_area_bin_counts).fillna(0) * 0.1
    )
    result["_english_local_score"] = (
        result.get("query_local_language", pd.Series(index=result.index))
        .eq("en")
        .astype(int)
    )
    result["_knowledge_graph_score"] = (
        result["osm_tags"].apply(_has_osm_knowledge_graph_tag)
        if "osm_tags" in result.columns
        else 0
    )
    result["_latin_name_score"] = result["polygon_name"].apply(_latin_name_score)
    result["_high_yield_name_score"] = result["polygon_name"].apply(
        _high_yield_place_name_score
    )
    result["_name_quality"] = result["polygon_name"].apply(_name_quality_score)

    result = result.sort_values(
        [
            "_high_yield_name_score",
            "_knowledge_graph_score",
            "_english_local_score",
            "_latin_name_score",
            "_region_score",
            "_area_bin_score",
            "_name_quality",
            "world_region",
            "area_size_bin",
            "polygon_name",
        ],
        ascending=[False, False, False, False, True, True, False, True, True, True],
    )

    return result.drop(
        columns=[
            "_region_score",
            "_area_bin_score",
            "_english_local_score",
            "_knowledge_graph_score",
            "_latin_name_score",
            "_high_yield_name_score",
            "_name_quality",
        ]
    ).reset_index(drop=True)


def load_completion_source_polygons() -> pd.DataFrame:
    source_gdf = load_geodataframe(SOURCE_POLYGONS_PATH)
    if _metadata_ready(source_gdf):
        return source_gdf

    from georeset_osm_web_evidence.evidence.worldwide_pilot import add_pilot_metadata

    return add_pilot_metadata(source_gdf)


def _append_unique_rows(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    return (
        pd.concat([existing_df, new_df], ignore_index=True)
        .drop_duplicates(subset=subset, keep="first")
        .reset_index(drop=True)
    )


def _candidate_temp_path(polygon_row) -> Path:
    return OUTPUT_DIR / f"_tmp_page_text_{polygon_row.osm_type}_{polygon_row.osm_id}.parquet"


def _empty_search_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "osm_type",
            "osm_id",
            "polygon_name",
            "has_wikipedia_articles",
            "query",
            "provider",
            "rank",
            "title",
            "url",
            "description",
            "query_language",
            "world_region",
            "country",
            "local_language",
            "query_local_language",
            "area_size_bin",
            "polygon_category",
        ]
    )


def _empty_search_attempts() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "osm_type",
            "osm_id",
            "polygon_name",
            "has_wikipedia_articles",
            "query",
            "attempted_at",
            "result_count",
            "query_language",
            "search_error",
        ]
    )


def search_one_polygon(
    polygon_row,
    search_func: Callable[..., list[dict]] = search_brave,
    sleep_func: Callable[[float], None] = time.sleep,
    results_per_query: int = RESULTS_PER_QUERY,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = build_location_topic_english_search_queries(
        osm_tags=polygon_row.osm_tags,
        country=polygon_row.country,
        world_region=polygon_row.world_region,
        source_extract_id=polygon_row.source_extract_id,
        polygon_category=polygon_row.polygon_category,
        max_queries=MAX_QUERIES_PER_POLYGON,
    )
    rows = []
    attempt_rows = []

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
        else _empty_search_results()
    )
    search_attempts_df = (
        pd.read_parquet(SEARCH_ATTEMPTS_PATH)
        if SEARCH_ATTEMPTS_PATH.exists()
        else _empty_search_attempts()
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

    while len(_polygon_keys(complete_df)) < target_polygon_count:
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
            len(_polygon_keys(complete_df)) + 1,
            target_polygon_count,
            polygon_row.polygon_name,
            polygon_row.world_region,
            polygon_row.area_size_bin,
        )

        new_results_df, new_attempts_df = search_one_polygon(
            polygon_row,
            logger=logger,
        )
        search_results_df = _append_unique_rows(
            search_results_df,
            new_results_df,
            subset=["osm_type", "osm_id", "query", "url"],
        )
        search_attempts_df = _append_unique_rows(
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
    FINAL_ANALYSIS_PATH.write_text(json.dumps(final_analysis, indent=2, sort_keys=True))

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
    FINAL_ANALYSIS_PATH.write_text(json.dumps(final_analysis, indent=2, sort_keys=True))

    return final_analysis


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

    return sentence_artifact_respects_sampling_limits(
        sentence_df,
        sentences_per_polygon=SENTENCES_PER_POLYGON,
        sentences_per_url=SENTENCES_PER_URL,
        target_polygon_count=TARGET_POLYGON_COUNT,
    )


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
        FINAL_SEARCH_RESULTS_PATH,
        FINAL_CANDIDATE_URLS_PATH,
        FINAL_PAGE_TEXT_PATH,
        FINAL_PAGE_TEXT_WITH_QUALITY_PATH,
        ANALYSIS_PATH,
        FINAL_ANALYSIS_PATH,
    ]:
        path.unlink(missing_ok=True)


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

    prompt_df = run_location_topic_labeling_request_build()
    logger.info("LLM prompt rows: %s", len(prompt_df))
    final_analysis = finalize_existing_location_topic_outputs(
        urls_per_polygon=SENTENCES_PER_POLYGON,
    )
    logger.info("Final exact URL analysis: %s", json.dumps(final_analysis, sort_keys=True))

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
    write_analysis(analysis, logger)
    logger.info("English location-topic sentence pilot finished")


if __name__ == "__main__":
    main()

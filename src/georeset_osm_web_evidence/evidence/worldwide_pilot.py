import pandas as pd

from georeset_osm_web_evidence.osm.worldwide_balancing import compute_group_targets
from georeset_osm_web_evidence.search.languages import resolve_query_local_language
from georeset_osm_web_evidence.search.queries import (
    build_search_queries,
    classify_polygon,
    get_osm_name,
)
from georeset_osm_web_evidence.search.results import attempt_to_row, result_to_row


AREA_BIN_ORDER = ["tiny", "small", "medium", "large"]
POLYGON_KEY = ["osm_type", "osm_id"]
POLYGON_METADATA_COLUMNS = [
    "osm_type",
    "osm_id",
    "world_region",
    "country",
    "local_language",
    "query_local_language",
    "area_size_bin",
    "polygon_category",
]


def add_pilot_metadata(polygons_df: pd.DataFrame) -> pd.DataFrame:
    result = polygons_df.copy()
    result["polygon_name"] = result["osm_tags"].apply(get_osm_name)
    result["polygon_category"] = result["osm_tags"].apply(classify_polygon)
    result["query_local_language"] = result.apply(resolve_query_local_language, axis=1)
    result["has_wikipedia_articles"] = pd.NA

    return result


def query_languages_for_local_language(
    local_language: str | None,
    supported_languages: set[str],
) -> tuple[str, ...]:
    languages = ["en"]
    if (
        isinstance(local_language, str)
        and local_language != "en"
        and local_language in supported_languages
    ):
        languages.append(local_language)

    return tuple(languages)


def build_limited_localized_queries(
    osm_tags: dict,
    local_language: str | None,
    supported_languages: set[str],
    max_queries: int,
) -> list[tuple[str, str]]:
    query_languages = query_languages_for_local_language(
        local_language=local_language,
        supported_languages=supported_languages,
    )
    queries_by_language = {
        language: build_search_queries(osm_tags, search_languages=(language,))
        for language in query_languages
    }
    max_language_query_count = max(
        (len(queries) for queries in queries_by_language.values()),
        default=0,
    )
    interleaved_queries = []

    for query_index in range(max_language_query_count):
        for language in query_languages:
            language_queries = queries_by_language[language]
            if query_index < len(language_queries):
                interleaved_queries.append((language, language_queries[query_index]))

    return interleaved_queries[:max_queries]


def _as_dataframe_like(input_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(input_df, "geometry") and "geometry" in result_df.columns:
        import geopandas as gpd

        return gpd.GeoDataFrame(
            result_df,
            geometry=input_df.geometry.name,
            crs=input_df.crs,
        )

    return result_df


def _ordered_area_bins(df: pd.DataFrame) -> list[str]:
    area_bins = [area_bin for area_bin in AREA_BIN_ORDER if area_bin in set(df["area_size_bin"])]
    extra_bins = sorted(set(df["area_size_bin"]) - set(area_bins))

    return area_bins + extra_bins


def select_stratified_pilot_polygons(
    polygons_df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if polygons_df.empty or sample_size <= 0:
        return polygons_df.head(0).copy()

    group_columns = [
        column
        for column in ["world_region", "area_size_bin"]
        if column in polygons_df.columns
    ]
    if not group_columns:
        result = polygons_df.sample(
            n=min(sample_size, len(polygons_df)),
            random_state=random_state,
        )
        return _as_dataframe_like(polygons_df, result.reset_index(drop=True))

    region_targets = compute_group_targets(
        polygons_df,
        sample_size=min(sample_size, len(polygons_df)),
        group_columns=["world_region"] if "world_region" in group_columns else group_columns,
    )
    area_bins = _ordered_area_bins(polygons_df) if "area_size_bin" in group_columns else []
    selected_parts = []
    selected_indices = set()

    for region_index, (region_key, target) in enumerate(sorted(region_targets.items())):
        if "world_region" in group_columns:
            region_value = region_key[0]
            region_df = polygons_df[polygons_df["world_region"] == region_value]
        else:
            region_df = polygons_df

        for target_index in range(target):
            candidate_df = region_df.drop(index=list(selected_indices), errors="ignore")
            if candidate_df.empty:
                continue

            if area_bins:
                rotated_bins = (
                    area_bins[(region_index + target_index) % len(area_bins):]
                    + area_bins[:(region_index + target_index) % len(area_bins)]
                )
                for area_bin in rotated_bins:
                    area_df = candidate_df[candidate_df["area_size_bin"] == area_bin]
                    if not area_df.empty:
                        candidate_df = area_df
                        break

            row = candidate_df.sample(
                n=1,
                random_state=random_state + region_index + target_index,
            )
            selected_parts.append(row)
            selected_indices.update(row.index.to_list())

    selected_df = (
        pd.concat(selected_parts)
        if selected_parts
        else polygons_df.head(0).copy()
    )

    if len(selected_df) < min(sample_size, len(polygons_df)):
        remaining_df = polygons_df.drop(index=list(selected_indices), errors="ignore")
        fill_count = min(sample_size, len(polygons_df)) - len(selected_df)
        fill_df = remaining_df.sample(n=fill_count, random_state=random_state)
        selected_df = pd.concat([selected_df, fill_df])

    return _as_dataframe_like(polygons_df, selected_df.reset_index(drop=True))


def _combine_unique_strings(values) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def build_candidate_urls(
    search_results_df: pd.DataFrame,
    max_urls_per_polygon: int | None = None,
) -> pd.DataFrame:
    if search_results_df.empty:
        return pd.DataFrame(
            columns=[
                "osm_type",
                "osm_id",
                "polygon_name",
                "has_wikipedia_articles",
                "provider",
                "url",
                "best_rank",
                "title",
                "description",
                "queries",
            ]
        )

    candidate_urls_df = (
        search_results_df.sort_values("rank")
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
            dropna=False,
        )
        .agg(
            best_rank=("rank", "min"),
            title=("title", "first"),
            description=("description", "first"),
            queries=("query", _combine_unique_strings),
        )
        .sort_values(["osm_type", "osm_id", "best_rank", "url"])
        .reset_index(drop=True)
    )

    if max_urls_per_polygon is not None:
        candidate_urls_df = (
            candidate_urls_df.groupby(POLYGON_KEY, group_keys=False, dropna=False)
            .head(max_urls_per_polygon)
            .reset_index(drop=True)
        )

    return candidate_urls_df


def build_search_rows_for_query(
    polygon_row,
    query_language: str,
    query: str,
    results: list[dict],
    search_error: str | None,
) -> tuple[list[dict], dict]:
    attempt_row = attempt_to_row(
        polygon_row=polygon_row,
        polygon_name=polygon_row.polygon_name,
        query=query,
        result_count=len(results),
    )
    attempt_row.update({
        "query_language": query_language,
        "search_error": search_error,
    })

    result_rows = []
    for rank, result in enumerate(results, start=1):
        result_row = result_to_row(
            polygon_row=polygon_row,
            polygon_name=polygon_row.polygon_name,
            query=query,
            rank=rank,
            result=result,
        )
        result_row.update({
            "query_language": query_language,
            "world_region": polygon_row.world_region,
            "country": polygon_row.country,
            "local_language": polygon_row.local_language,
            "query_local_language": polygon_row.query_local_language,
            "area_size_bin": polygon_row.area_size_bin,
            "polygon_category": polygon_row.polygon_category,
        })
        result_rows.append(result_row)

    return result_rows, attempt_row


def attach_polygon_metadata(
    df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in POLYGON_METADATA_COLUMNS
        if column in pilot_gdf.columns
    ]
    replaceable_metadata_columns = [
        column
        for column in metadata_columns
        if column not in POLYGON_KEY and column in df.columns
    ]
    base_df = df.drop(columns=replaceable_metadata_columns)
    metadata_df = pilot_gdf[metadata_columns].drop_duplicates(POLYGON_KEY)

    return base_df.merge(
        metadata_df,
        on=POLYGON_KEY,
        how="left",
    )


def _unique_polygon_count(df: pd.DataFrame) -> int:
    key_columns = [column for column in POLYGON_KEY if column in df.columns]
    if not key_columns or df.empty:
        return 0

    return len(df[key_columns].drop_duplicates())


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {}

    return {
        str(key): int(value)
        for key, value in df[column].value_counts().sort_index().items()
    }


def summarize_sentence_pilot(
    polygons_df: pd.DataFrame,
    search_results_df: pd.DataFrame,
    candidate_urls_df: pd.DataFrame,
    page_text_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
) -> dict:
    successful_fetch_count = (
        int(page_text_df["fetch_error"].isna().sum())
        if "fetch_error" in page_text_df.columns
        else 0
    )
    high_quality_page_count = (
        int((page_text_df["quality_score"] >= 0.8).sum())
        if "quality_score" in page_text_df.columns
        else 0
    )

    return {
        "polygon_count": int(len(polygons_df)),
        "world_region_counts": _value_counts_dict(polygons_df, "world_region"),
        "area_size_bin_counts": _value_counts_dict(polygons_df, "area_size_bin"),
        "search_result_count": int(len(search_results_df)),
        "candidate_url_count": int(len(candidate_urls_df)),
        "fetched_url_count": int(len(page_text_df)),
        "successful_fetch_count": successful_fetch_count,
        "high_quality_page_count": high_quality_page_count,
        "sentence_count": int(len(sentence_df)),
        "polygons_with_sentences": _unique_polygon_count(sentence_df),
    }

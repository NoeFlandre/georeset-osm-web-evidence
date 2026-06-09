from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.search.queries import build_search_queries, get_osm_name


POLYGON_KEY = ["osm_type", "osm_id"]
QUERY_KEY = POLYGON_KEY + ["query"]
SEARCH_ATTEMPT_COLUMNS = QUERY_KEY + [
    "polygon_name",
    "has_wikipedia_articles",
    "attempted_at",
    "result_count",
]


def _require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{path} is missing required columns: {missing_text}")


def load_existing_search_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=QUERY_KEY)

    return pd.read_parquet(path)


def load_existing_search_attempts(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=SEARCH_ATTEMPT_COLUMNS)

    attempts_df = pd.read_parquet(path)
    _require_columns(attempts_df, SEARCH_ATTEMPT_COLUMNS, path)

    return attempts_df


def polygon_keys(df: pd.DataFrame) -> pd.DataFrame:
    return df[POLYGON_KEY].drop_duplicates()


def query_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "query" not in df.columns:
        return pd.DataFrame(columns=QUERY_KEY)

    return df[QUERY_KEY].drop_duplicates()


def build_expected_query_table(
    polygons_df: pd.DataFrame,
    search_languages: list[str] | tuple[str, ...] = ("fr",),
) -> pd.DataFrame:
    rows = []
    columns = POLYGON_KEY + ["polygon_name", "has_wikipedia_articles", "query"]

    for polygon_row in polygons_df.itertuples():
        polygon_name = get_osm_name(polygon_row.osm_tags)

        for query in build_search_queries(
            polygon_row.osm_tags,
            search_languages=search_languages,
        ):
            rows.append(
                {
                    "osm_type": polygon_row.osm_type,
                    "osm_id": polygon_row.osm_id,
                    "polygon_name": polygon_name,
                    "has_wikipedia_articles": polygon_row.has_wikipedia_articles,
                    "query": query,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def find_missing_queries(
    expected_queries_df: pd.DataFrame,
    search_results_df: pd.DataFrame,
    attempted_queries_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    completed_queries_df = query_keys(search_results_df)

    if attempted_queries_df is not None and not attempted_queries_df.empty:
        completed_queries_df = pd.concat(
            [completed_queries_df, query_keys(attempted_queries_df)],
            ignore_index=True,
        ).drop_duplicates()

    return expected_queries_df.merge(
        completed_queries_df,
        on=QUERY_KEY,
        how="left",
        indicator=True,
    ).query("_merge == 'left_only'").drop(columns=["_merge"])


def summarize_search_coverage(
    polygons_df: pd.DataFrame,
    search_results_df: pd.DataFrame,
    attempted_polygons_df: pd.DataFrame | None = None,
    search_languages: list[str] | tuple[str, ...] = ("fr",),
) -> dict:
    total_polygons = len(polygon_keys(polygons_df))
    searched_keys = polygon_keys(search_results_df)

    if attempted_polygons_df is not None and not attempted_polygons_df.empty:
        searched_keys = pd.concat(
            [searched_keys, polygon_keys(attempted_polygons_df)],
            ignore_index=True,
        ).drop_duplicates()

    searched_polygons = len(searched_keys)
    searched_queries = (
        len(search_results_df[POLYGON_KEY + ["query"]].drop_duplicates())
        if "query" in search_results_df.columns
        else 0
    )
    expected_queries = None
    missing_queries = None

    if "osm_tags" in polygons_df.columns:
        expected_queries_df = build_expected_query_table(
            polygons_df,
            search_languages=search_languages,
        )
        expected_queries = len(expected_queries_df)
        missing_queries = len(
            find_missing_queries(
                expected_queries_df,
                search_results_df,
                attempted_queries_df=attempted_polygons_df,
            )
        )

    summary = {
        "total_polygons": total_polygons,
        "searched_polygons": searched_polygons,
        "unsearched_polygons": total_polygons - searched_polygons,
        "searched_queries": searched_queries,
    }

    if expected_queries is not None:
        summary["expected_queries"] = expected_queries
        summary["missing_queries"] = missing_queries

    return summary


def unsearched_polygons(
    polygons_df: pd.DataFrame,
    search_results_df: pd.DataFrame,
    attempted_polygons_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    searched_keys = polygon_keys(search_results_df)

    if attempted_polygons_df is not None and not attempted_polygons_df.empty:
        searched_keys = pd.concat(
            [searched_keys, polygon_keys(attempted_polygons_df)],
            ignore_index=True,
        ).drop_duplicates()

    return polygons_df.merge(
        searched_keys,
        on=POLYGON_KEY,
        how="left",
        indicator=True,
    ).query("_merge == 'left_only'").drop(columns=["_merge"])


def choose_unsearched_polygons(
    polygons_df: pd.DataFrame,
    search_results_df: pd.DataFrame,
    polygon_limit: int,
    attempted_polygons_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidates_df = unsearched_polygons(
        polygons_df,
        search_results_df,
        attempted_polygons_df=attempted_polygons_df,
    )

    if candidates_df.empty:
        return candidates_df

    if "has_wikipedia_articles" not in candidates_df.columns:
        return candidates_df.head(polygon_limit)

    per_status_limit = max(1, polygon_limit // 2)
    selected_parts = []

    for has_wikipedia_articles in [False, True]:
        status_df = candidates_df[
            candidates_df["has_wikipedia_articles"] == has_wikipedia_articles
        ]
        selected_parts.append(status_df.head(per_status_limit))

    selected_df = pd.concat(selected_parts, ignore_index=True)

    if len(selected_df) < polygon_limit:
        already_selected = polygon_keys(selected_df)
        remaining_df = candidates_df.merge(
            already_selected,
            on=POLYGON_KEY,
            how="left",
            indicator=True,
        ).query("_merge == 'left_only'").drop(columns=["_merge"])
        selected_df = pd.concat(
            [selected_df, remaining_df.head(polygon_limit - len(selected_df))],
            ignore_index=True,
        )

    return selected_df.head(polygon_limit)


def choose_polygons_to_search(
    polygons_df: pd.DataFrame,
    existing_results_df: pd.DataFrame,
    existing_attempts_df: pd.DataFrame,
    new_polygon_limit: int,
    complete_existing_polygons_only: bool,
) -> pd.DataFrame:
    if not complete_existing_polygons_only:
        return choose_unsearched_polygons(
            polygons_df,
            existing_results_df,
            polygon_limit=new_polygon_limit,
            attempted_polygons_df=existing_attempts_df,
        )

    treated_parts = []
    for dataframe in [existing_results_df, existing_attempts_df]:
        if not dataframe.empty:
            treated_parts.append(polygon_keys(dataframe))

    if not treated_parts:
        return polygons_df.head(0).copy()

    treated_keys = pd.concat(treated_parts, ignore_index=True).drop_duplicates()

    return polygons_df.merge(treated_keys, on=POLYGON_KEY, how="inner")

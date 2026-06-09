import unicodedata

import pandas as pd


POLYGON_KEY_COLUMNS = ["osm_type", "osm_id"]


def polygon_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=POLYGON_KEY_COLUMNS)

    return df[POLYGON_KEY_COLUMNS].drop_duplicates()


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
    complete_keys_df = polygon_keys(complete_df)
    attempted_keys_df = polygon_keys(attempted_df)
    excluded_keys_df = pd.concat(
        [complete_keys_df, attempted_keys_df],
        ignore_index=True,
    ).drop_duplicates()

    if excluded_keys_df.empty:
        remaining_df = source_df.copy()
    else:
        remaining_df = source_df.merge(
            excluded_keys_df.assign(_exclude=True),
            on=POLYGON_KEY_COLUMNS,
            how="left",
        )
        remaining_df = remaining_df[remaining_df["_exclude"].isna()].drop(
            columns=["_exclude"]
        )

    region_counts = complete_df["world_region"].value_counts().to_dict()
    area_bin_counts = complete_df["area_size_bin"].value_counts().to_dict()
    attempted_metadata_df = polygon_keys(attempted_df).merge(
        source_df[POLYGON_KEY_COLUMNS + ["world_region", "area_size_bin"]],
        on=POLYGON_KEY_COLUMNS,
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

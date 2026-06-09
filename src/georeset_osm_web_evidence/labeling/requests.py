import json
import hashlib
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.prompt import (
    LOCATION_AWARE_PROMPT_VERSION,
    PROMPT_VERSION,
    build_binary_label_prompt,
    build_location_aware_binary_label_prompt,
)

LABELING_OUTPUT_COLUMNS = [
    "sentence_id",
    "model_input",
    "prompt_version",
    "prompt",
    "llm_label",
    "raw_response",
    "parse_error",
]


def normalize_model_input(text: str) -> str:
    return " ".join(text.split())


def make_sentence_candidate_id(row: pd.Series) -> str:
    payload = "|".join(
        str(row.get(column, ""))
        for column in ["osm_type", "osm_id", "url", "sentence"]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_labeling_prompt_rows(
    labeling_df: pd.DataFrame,
    limit: int | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> pd.DataFrame:
    result = labeling_df.copy()

    if limit is not None:
        result = result.head(limit).copy()

    result["prompt_version"] = prompt_version
    result["prompt"] = result["model_input"].apply(build_binary_label_prompt)
    result["llm_label"] = None
    result["raw_response"] = None
    result["parse_error"] = None

    metadata_columns = [
        column for column in result.columns if column not in LABELING_OUTPUT_COLUMNS
    ]
    ordered_columns = LABELING_OUTPUT_COLUMNS + metadata_columns

    return result.loc[:, ordered_columns].reset_index(drop=True)


def build_sentence_candidate_prompt_rows(
    sentence_df: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    result = sentence_df.copy()
    result = result[result["sentence"].apply(lambda sentence: isinstance(sentence, str))]
    result["model_input"] = result["sentence"].apply(normalize_model_input)
    result = result[result["model_input"] != ""].copy()
    result["sentence_id"] = result.apply(make_sentence_candidate_id, axis=1)

    return build_labeling_prompt_rows(result, limit=limit)


def _join_location_context(row: pd.Series) -> str:
    values = []

    for column in ["country", "world_region"]:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            values.append(" ".join(value.split()))

    return ", ".join(dict.fromkeys(values))


def build_location_aware_sentence_candidate_prompt_rows(
    sentence_df: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    result = sentence_df.copy()
    result = result[result["sentence"].apply(lambda sentence: isinstance(sentence, str))]
    result["model_input"] = result["sentence"].apply(normalize_model_input)
    result = result[result["model_input"] != ""].copy()
    result["sentence_id"] = result.apply(make_sentence_candidate_id, axis=1)

    if limit is not None:
        result = result.head(limit).copy()

    prompts = []
    for _, row in result.iterrows():
        prompts.append(
            build_location_aware_binary_label_prompt(
                sentence=row["model_input"],
                polygon_name=row["polygon_name"],
                location_context=_join_location_context(row),
                polygon_category=row.get("polygon_category", ""),
                page_title=row.get("title", ""),
                search_query=row.get("search_queries", ""),
            )
        )

    result["prompt_version"] = LOCATION_AWARE_PROMPT_VERSION
    result["prompt"] = prompts
    result["llm_label"] = None
    result["raw_response"] = None
    result["parse_error"] = None

    metadata_columns = [
        column for column in result.columns if column not in LABELING_OUTPUT_COLUMNS
    ]
    ordered_columns = LABELING_OUTPUT_COLUMNS + metadata_columns

    return result.loc[:, ordered_columns].reset_index(drop=True)


def write_labeling_prompt_jsonl(
    prompt_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for _, row in prompt_df.iterrows():
            payload = {
                "sentence_id": row["sentence_id"],
                "prompt_version": row["prompt_version"],
                "prompt": row["prompt"],
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

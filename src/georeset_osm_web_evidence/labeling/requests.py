import json
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.prompt import (
    PROMPT_VERSION,
    build_binary_label_prompt,
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

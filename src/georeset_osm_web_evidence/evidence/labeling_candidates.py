import hashlib
import json
from pathlib import Path

import pandas as pd


def normalize_model_input(text: str) -> str:
    return " ".join(text.split())


def make_sentence_id(model_input: str) -> str:
    return hashlib.sha256(model_input.encode("utf-8")).hexdigest()


def build_labeling_candidates(
    sentence_df: pd.DataFrame,
    min_quality_score: float = 0.8,
) -> pd.DataFrame:
    result = sentence_df.copy()
    result = result[result["quality_score"] >= min_quality_score]
    result = result[result["sentence"].apply(lambda sentence: isinstance(sentence, str))]

    result["model_input"] = result["sentence"].apply(normalize_model_input)
    result = result[result["model_input"] != ""]
    result = result.drop_duplicates(subset=["model_input"], keep="first")
    result["sentence_id"] = result["model_input"].apply(make_sentence_id)

    ordered_columns = ["sentence_id", "model_input"] + [
        column
        for column in result.columns
        if column not in {"sentence_id", "model_input"}
    ]

    return result.loc[:, ordered_columns].reset_index(drop=True)


def write_labeling_candidates_jsonl(
    labeling_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for _, row in labeling_df.iterrows():
            payload = {
                "sentence_id": row["sentence_id"],
                "text": row["model_input"],
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

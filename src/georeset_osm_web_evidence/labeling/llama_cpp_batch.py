from collections.abc import Callable
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.llama_cpp import (
    create_llama_cpp_label_fn,
    get_llama_cpp_model_settings_from_env,
    load_llama_cpp_model,
)
from georeset_osm_web_evidence.labeling.runner import label_prompt_rows
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact


def format_llm_labeling_summary(
    labeled_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    parts = [
        f"Saved {len(labeled_df)} LLM-labeled rows to {Path(output_path)}",
        str(labeled_df["llm_label"].value_counts(dropna=False)),
    ]
    if "parse_error" in labeled_df.columns:
        parts.append(str(labeled_df["parse_error"].value_counts(dropna=False)))

    return "\n".join(parts)


def run_llama_cpp_prompt_batch(
    input_path: str | Path,
    output_path: str | Path,
    label_fn: Callable[[str], str] | None = None,
    *,
    row_limit: int | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    prompt_df = pd.read_parquet(input_path).copy()
    if row_limit is not None:
        prompt_df = prompt_df.head(row_limit).copy()

    if label_fn is None:
        model_settings = get_llama_cpp_model_settings_from_env()
        llm = load_llama_cpp_model(
            repo_id=model_settings["repo_id"],
            filename=model_settings["filename"],
            chat_template_kwargs=model_settings["chat_template_kwargs"],
            **model_settings["model_kwargs"],
        )
        label_fn = create_llama_cpp_label_fn(llm)

    labeled_df = label_prompt_rows(prompt_df, label_fn)

    write_dataframe_artifact(labeled_df, output_path)

    return labeled_df

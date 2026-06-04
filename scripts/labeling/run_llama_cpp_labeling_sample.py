from collections.abc import Callable
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.llama_cpp import (
    create_llama_cpp_label_fn,
    get_llama_cpp_model_settings_from_env,
    load_llama_cpp_model,
)
from georeset_osm_web_evidence.labeling.runner import label_prompt_rows

DEFAULT_INPUT_PATH = Path(
    "data/processed/labeling/llm_labeling_requests_sample.parquet"
)
DEFAULT_OUTPUT_PATH = Path("data/processed/labeling/llm_labels_sample.parquet")


def run_llama_cpp_labeling_sample(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sample_size: int = 5,
    label_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    prompt_df = pd.read_parquet(input_path).head(sample_size).copy()

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(output_path, index=False)

    return labeled_df


def main() -> None:
    labeled_df = run_llama_cpp_labeling_sample()

    print(f"Saved {len(labeled_df)} LLM-labeled rows to {DEFAULT_OUTPUT_PATH}")
    print(labeled_df["llm_label"].value_counts(dropna=False))
    if "parse_error" in labeled_df.columns:
        print(labeled_df["parse_error"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.requests import (
    build_and_write_labeling_prompt_artifacts,
    build_labeling_prompt_rows,
)

DEFAULT_INPUT_PATH = Path("data/processed/evidence/labeling_candidates.parquet")
DEFAULT_PARQUET_OUTPUT_PATH = Path(
    "data/processed/labeling/llm_labeling_requests_sample.parquet"
)
DEFAULT_JSONL_OUTPUT_PATH = Path(
    "data/processed/labeling/llm_labeling_requests_sample.jsonl"
)


def run_labeling_prompt_sample_build(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    parquet_output_path: str | Path = DEFAULT_PARQUET_OUTPUT_PATH,
    jsonl_output_path: str | Path = DEFAULT_JSONL_OUTPUT_PATH,
    sample_size: int = 20,
) -> pd.DataFrame:
    return build_and_write_labeling_prompt_artifacts(
        input_path=input_path,
        parquet_output_path=parquet_output_path,
        jsonl_output_path=jsonl_output_path,
        prompt_builder=lambda labeling_df: build_labeling_prompt_rows(
            labeling_df,
            limit=sample_size,
        ),
    )


def main() -> None:
    prompt_df = run_labeling_prompt_sample_build()

    print(
        "Saved "
        f"{len(prompt_df)} LLM labeling request rows to {DEFAULT_PARQUET_OUTPUT_PATH}"
    )
    print(f"Saved JSONL prompts to {DEFAULT_JSONL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

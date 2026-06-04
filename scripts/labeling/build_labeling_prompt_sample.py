from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.requests import (
    build_labeling_prompt_rows,
    write_labeling_prompt_jsonl,
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
    input_path = Path(input_path)
    parquet_output_path = Path(parquet_output_path)
    jsonl_output_path = Path(jsonl_output_path)

    labeling_df = pd.read_parquet(input_path)
    prompt_df = build_labeling_prompt_rows(labeling_df, limit=sample_size)

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_df.to_parquet(parquet_output_path, index=False)
    write_labeling_prompt_jsonl(prompt_df, jsonl_output_path)

    return prompt_df


def main() -> None:
    prompt_df = run_labeling_prompt_sample_build()

    print(
        "Saved "
        f"{len(prompt_df)} LLM labeling request rows to {DEFAULT_PARQUET_OUTPUT_PATH}"
    )
    print(f"Saved JSONL prompts to {DEFAULT_JSONL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

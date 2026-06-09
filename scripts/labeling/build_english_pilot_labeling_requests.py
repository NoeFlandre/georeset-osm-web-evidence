from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.requests import (
    build_and_write_labeling_prompt_artifacts,
    build_sentence_candidate_prompt_rows,
)

DEFAULT_INPUT_PATH = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only/"
    "sentence_candidates.parquet"
)
DEFAULT_PARQUET_OUTPUT_PATH = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only/"
    "llm_labeling_requests.parquet"
)
DEFAULT_JSONL_OUTPUT_PATH = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only/"
    "llm_labeling_requests.jsonl"
)


def run_english_pilot_labeling_request_build(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    parquet_output_path: str | Path = DEFAULT_PARQUET_OUTPUT_PATH,
    jsonl_output_path: str | Path = DEFAULT_JSONL_OUTPUT_PATH,
) -> pd.DataFrame:
    return build_and_write_labeling_prompt_artifacts(
        input_path=input_path,
        parquet_output_path=parquet_output_path,
        jsonl_output_path=jsonl_output_path,
        prompt_builder=build_sentence_candidate_prompt_rows,
    )


def main() -> None:
    prompt_df = run_english_pilot_labeling_request_build()

    print(
        "Saved "
        f"{len(prompt_df)} English-pilot LLM request rows to "
        f"{DEFAULT_PARQUET_OUTPUT_PATH}"
    )
    print(f"Saved JSONL prompts to {DEFAULT_JSONL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

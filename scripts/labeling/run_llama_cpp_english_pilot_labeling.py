from collections.abc import Callable
from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.labeling.llama_cpp_batch import (
    format_llm_labeling_summary,
    run_llama_cpp_prompt_batch,
)

DEFAULT_INPUT_PATH = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only/"
    "llm_labeling_requests.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only/"
    "sentence_candidates_llm_labeled.parquet"
)


def run_llama_cpp_english_pilot_labeling(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    label_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    return run_llama_cpp_prompt_batch(
        input_path=input_path,
        output_path=output_path,
        label_fn=label_fn,
    )


def main() -> None:
    labeled_df = run_llama_cpp_english_pilot_labeling()

    print(format_llm_labeling_summary(labeled_df, DEFAULT_OUTPUT_PATH))


if __name__ == "__main__":
    main()

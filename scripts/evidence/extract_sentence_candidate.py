from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.sentence_candidates import (
    build_sentence_candidate_dataframe,
)
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact

DEFAULT_INPUT_PATH = Path(
    "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
)
DEFAULT_OUTPUT_PATH = Path("data/processed/evidence/sentence_candidates.parquet")


def run_sentence_candidate_build(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    text_df = pd.read_parquet(input_path)
    sentence_df = build_sentence_candidate_dataframe(text_df)
    return write_dataframe_artifact(sentence_df, output_path)


def main():
    sentence_df = run_sentence_candidate_build(
        input_path=DEFAULT_INPUT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
    )
    print(
        f"Extracted {len(sentence_df)} sentences from {sentence_df['url'].nunique()} URLs"
    )
    print(f"Saved the sentence dataframe to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

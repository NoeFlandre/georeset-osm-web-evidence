from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.labeling_candidates import (
    build_labeling_candidates,
    write_labeling_candidates_jsonl,
)

DEFAULT_INPUT_PATH = Path("data/processed/evidence/sentence_candidates.parquet")
DEFAULT_PARQUET_OUTPUT_PATH = Path(
    "data/processed/evidence/labeling_candidates.parquet"
)
DEFAULT_JSONL_OUTPUT_PATH = Path("data/processed/evidence/labeling_candidates.jsonl")


def run_labeling_candidate_build(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    parquet_output_path: str | Path = DEFAULT_PARQUET_OUTPUT_PATH,
    jsonl_output_path: str | Path = DEFAULT_JSONL_OUTPUT_PATH,
    min_quality_score: float = 0.8,
) -> pd.DataFrame:
    input_path = Path(input_path)
    parquet_output_path = Path(parquet_output_path)
    jsonl_output_path = Path(jsonl_output_path)

    sentence_df = pd.read_parquet(input_path)
    labeling_df = build_labeling_candidates(
        sentence_df,
        min_quality_score=min_quality_score,
    )

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    labeling_df.to_parquet(parquet_output_path, index=False)
    write_labeling_candidates_jsonl(labeling_df, jsonl_output_path)

    return labeling_df


def print_labeling_candidate_summary(
    labeling_df: pd.DataFrame,
    parquet_output_path: Path,
    jsonl_output_path: Path,
) -> None:
    print(f"Saved {len(labeling_df)} labeling candidates to {parquet_output_path}")
    print(f"Saved JSONL inputs to {jsonl_output_path}")

    if {"osm_type", "osm_id"}.issubset(labeling_df.columns):
        polygon_count = labeling_df[["osm_type", "osm_id"]].drop_duplicates().shape[0]
        print(f"Covered {polygon_count} polygons")

    if "url" in labeling_df.columns:
        print(f"Covered {labeling_df['url'].nunique()} URLs")

    if "quality_score" in labeling_df.columns and not labeling_df.empty:
        print(f"Mean quality score: {labeling_df['quality_score'].mean():.3f}")


def main() -> None:
    labeling_df = run_labeling_candidate_build(
        input_path=DEFAULT_INPUT_PATH,
        parquet_output_path=DEFAULT_PARQUET_OUTPUT_PATH,
        jsonl_output_path=DEFAULT_JSONL_OUTPUT_PATH,
        min_quality_score=0.8,
    )
    print_labeling_candidate_summary(
        labeling_df,
        DEFAULT_PARQUET_OUTPUT_PATH,
        DEFAULT_JSONL_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

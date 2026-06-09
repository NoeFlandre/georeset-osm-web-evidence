from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.sample_sentence_candidates import (
    sample_sentence_candidates,
)
from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact

DEFAULT_INPUT_PATH = Path("data/processed/evidence/sentence_candidates.parquet")
DEFAULT_OUTPUT_PATH = Path(
    "data/processed/evidence/sentence_candidates_sample_200.parquet"
)


def run_sentence_candidate_sampling(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sample_size: int = 200,
    min_quality_score: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    sentence_df = pd.read_parquet(input_path)
    sampled_df = sample_sentence_candidates(
        sentence_df,
        sample_size=sample_size,
        min_quality_score=min_quality_score,
        random_state=random_state,
    ).reset_index(drop=True)

    write_dataframe_artifact(sampled_df, output_path)

    return sampled_df


def format_sampling_summary(
    sampled_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    lines = [f"Saved {len(sampled_df)} sampled sentences to {Path(output_path)}"]

    if {"osm_type", "osm_id"}.issubset(sampled_df.columns):
        polygon_count = sampled_df[["osm_type", "osm_id"]].drop_duplicates().shape[0]
        lines.append(f"Covered {polygon_count} polygons")

    if "has_wikipedia_articles" in sampled_df.columns:
        wikipedia_coverage = sampled_df["has_wikipedia_articles"].value_counts(
            dropna=False
        )
        lines.append("Wikipedia coverage:")
        lines.append(str(wikipedia_coverage))

    if "quality_score" in sampled_df.columns and not sampled_df.empty:
        lines.append(f"Mean quality score: {sampled_df['quality_score'].mean():.3f}")

    return "\n".join(lines)


def print_sampling_summary(sampled_df: pd.DataFrame, output_path: Path) -> None:
    print(format_sampling_summary(sampled_df, output_path))


def main() -> None:
    sampled_df = run_sentence_candidate_sampling(
        input_path=DEFAULT_INPUT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        sample_size=200,
        min_quality_score=0.8,
        random_state=42,
    )
    print_sampling_summary(sampled_df, DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    main()

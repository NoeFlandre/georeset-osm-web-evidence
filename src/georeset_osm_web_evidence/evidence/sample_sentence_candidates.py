import pandas as pd


def sample_sentence_candidates(
    sentence_df: pd.DataFrame,
    sample_size: int = 200,
    min_quality_score: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    high_quality_sentence_df = sentence_df[
        sentence_df["quality_score"] >= min_quality_score
    ]

    sample_size = min(sample_size, len(high_quality_sentence_df))

    sampled_sentence_df = high_quality_sentence_df.sample(
        n=sample_size, random_state=random_state
    )

    return sampled_sentence_df

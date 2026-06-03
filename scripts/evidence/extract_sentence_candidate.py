from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.evidence.sentence_candidates import (
    build_sentence_candidate_dataframe,
)


def main():
    input_path = Path(
        "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
    )
    output_path = Path("data/processed/evidence/sentence_candidates.parquet")

    text_df = pd.read_parquet(input_path)

    sentence_df = build_sentence_candidate_dataframe(text_df)

    sentence_df.to_parquet(output_path)
    print(
        f"Extracted {len(sentence_df)} sentences from {sentence_df['url'].nunique()} URLs"
    )
    print(f"Saved the sentence dataframe to {output_path}")


if __name__ == "__main__":
    main()

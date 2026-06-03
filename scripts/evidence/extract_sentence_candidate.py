from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.text.sentences import extract_sentence_candidates


def main():
    input_path = Path(
        "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
    )
    output_path = Path("data/processed/evidence/sentence_candidates.parquet")

    sentence_rows = []

    text_df = pd.read_parquet(input_path)

    for _, row in text_df.iterrows():
        text = row["text"]
        if not isinstance(text, str):
            continue

        sentence_candidates = extract_sentence_candidates(text)

        for sentence in sentence_candidates:
            sentence_rows.append(
                {
                    "osm_type": row["osm_type"],
                    "osm_id": row["osm_id"],
                    "polygon_name": row["polygon_name"],
                    "url": row["url"],
                    "sentence": sentence,
                }
            )

    sentence_df = pd.DataFrame(sentence_rows)
    sentence_df.to_parquet(output_path)
    print(f"Saved the sentence dataframe to {output_path}")


if __name__ == "__main__":
    main()

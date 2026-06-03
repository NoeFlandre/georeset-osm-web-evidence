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
                    "has_wikipedia_articles": row["has_wikipedia_articles"],
                    "url": row["url"],
                    "final_url": row["final_url"],
                    "search_title": row["search_title"],
                    "search_description": row["search_description"],
                    "search_queries": row["search_queries"],
                    "page_title": row["title"],
                    "text_length": row["text_length"],
                    "quality_score": row["quality_score"],
                    "quality_flags": row["quality_flags"],
                    "sentence": sentence,
                }
            )

    sentence_df = pd.DataFrame(sentence_rows)
    sentence_df.to_parquet(output_path)
    print(
        f"Extracted {len(sentence_df)} sentences from {sentence_df['url'].nunique()} URLs"
    )
    print(f"Saved the sentence dataframe to {output_path}")


if __name__ == "__main__":
    main()

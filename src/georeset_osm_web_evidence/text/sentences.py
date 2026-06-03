import re

import pandas as pd

from georeset_osm_web_evidence.text.utils import count_words


def split_sentences(text: str) -> list[str]:
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    return re.split(r"(?<=[.!?])\s+", clean_text)


def is_sentence_candidate(
    sentence: str, min_word_count: int = 8, max_word_count: int = 80
) -> bool:
    word_count = count_words(sentence)
    if word_count < min_word_count:
        return False
    if word_count > max_word_count:
        return False
    return True


def extract_sentence_candidates(
    text: str,
    min_word_count: int = 8,
    max_word_count: int = 80,
) -> list[str]:
    sentences = split_sentences(text)
    return [
        sentence
        for sentence in sentences
        if is_sentence_candidate(sentence, min_word_count, max_word_count)
    ]


def build_sentence_candidate_dataframe(text_df: pd.DataFrame) -> pd.DataFrame:
    sentence_rows = []

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
    return sentence_df

import pandas as pd

from georeset_osm_web_evidence.text.sentences import extract_sentence_candidates

SENTENCE_CANDIDATE_COLUMNS = [
    "osm_type",
    "osm_id",
    "polygon_name",
    "has_wikipedia_articles",
    "url",
    "final_url",
    "search_title",
    "search_description",
    "search_queries",
    "page_title",
    "text_length",
    "quality_score",
    "quality_flags",
    "sentence",
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

    sentence_df = pd.DataFrame(sentence_rows, columns=SENTENCE_CANDIDATE_COLUMNS)
    return sentence_df


def limit_sentence_candidates(
    sentence_df: pd.DataFrame,
    max_sentences_per_polygon: int,
    max_sentences_per_url: int,
) -> pd.DataFrame:
    if max_sentences_per_polygon <= 0:
        raise ValueError("max_sentences_per_polygon must be positive")
    if max_sentences_per_url <= 0:
        raise ValueError("max_sentences_per_url must be positive")
    if sentence_df.empty:
        return sentence_df.copy()

    polygon_columns = ["osm_type", "osm_id"]
    per_url_counts = sentence_df.groupby(
        polygon_columns + ["url"],
        dropna=False,
    ).cumcount()
    first_sentences_per_url_df = sentence_df[per_url_counts < max_sentences_per_url]

    per_polygon_counts = first_sentences_per_url_df.groupby(
        polygon_columns,
        dropna=False,
    ).cumcount()
    limited_df = first_sentences_per_url_df[
        per_polygon_counts < max_sentences_per_polygon
    ]

    return limited_df.reset_index(drop=True)


def select_complete_sentence_candidates(
    sentence_df: pd.DataFrame,
    sentences_per_polygon: int,
    sentences_per_url: int,
    target_polygon_count: int | None = None,
) -> pd.DataFrame:
    if target_polygon_count is not None and target_polygon_count <= 0:
        raise ValueError("target_polygon_count must be positive")

    limited_df = limit_sentence_candidates(
        sentence_df,
        max_sentences_per_polygon=sentences_per_polygon,
        max_sentences_per_url=sentences_per_url,
    )
    if limited_df.empty:
        return limited_df

    polygon_columns = ["osm_type", "osm_id"]
    sentence_counts = limited_df.groupby(
        polygon_columns,
        dropna=False,
    ).size()
    complete_polygon_keys = sentence_counts[
        sentence_counts == sentences_per_polygon
    ].index
    if target_polygon_count is not None:
        complete_polygon_keys = complete_polygon_keys[:target_polygon_count]

    complete_key_df = complete_polygon_keys.to_frame(index=False)
    return limited_df.merge(
        complete_key_df,
        on=polygon_columns,
        how="inner",
    ).reset_index(drop=True)

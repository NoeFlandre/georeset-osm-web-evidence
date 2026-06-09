import hashlib
import re

import justext
import pandas as pd

from georeset_osm_web_evidence.text.sentences import (
    SENTENCE_FILTER_PROFILE,
    SENTENCE_FILTER_RULES,
    extract_sentence_candidates,
)

ENGLISH_STOPWORDS = justext.get_stoplist("English")
NON_ENGLISH_STOPLISTS = [
    justext.get_stoplist(language)
    for language in ["Dutch", "French", "German", "Polish", "Portuguese", "Spanish"]
]
MINHASH_SIGNATURE_SIZE = 64
MINHASH_SHINGLE_SIZE = 5
MINHASH_DUPLICATE_THRESHOLD = 0.8
MINHASH_LSH_BAND_SIZE = 4

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
    "query_language",
    "sentence_filter_profile",
    "sentence_filter_rules",
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
                    "query_language": row.get("query_language", pd.NA),
                    "sentence_filter_profile": SENTENCE_FILTER_PROFILE,
                    "sentence_filter_rules": "; ".join(SENTENCE_FILTER_RULES),
                    "sentence": sentence,
                }
            )

    sentence_df = pd.DataFrame(sentence_rows, columns=SENTENCE_CANDIDATE_COLUMNS)
    return sentence_df


def looks_like_english_sentence(sentence: str) -> bool:
    tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", sentence.lower())
    if not tokens:
        return False

    english_count = sum(token in ENGLISH_STOPWORDS for token in tokens)
    non_english_count = max(
        sum(token in stoplist for token in tokens)
        for stoplist in NON_ENGLISH_STOPLISTS
    )

    return english_count >= 2 and english_count >= non_english_count


def filter_english_sentence_candidates(sentence_df: pd.DataFrame) -> pd.DataFrame:
    if sentence_df.empty:
        return sentence_df.copy()

    language_column = (
        "query_language"
        if "query_language" in sentence_df.columns
        else "query_local_language"
    )
    english_query_df = sentence_df[sentence_df[language_column].eq("en")]
    english_sentence_mask = english_query_df["sentence"].map(
        looks_like_english_sentence
    )
    return english_query_df[english_sentence_mask].reset_index(drop=True)


def tokenize_for_minhash(sentence: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", sentence.lower())


def build_word_shingles(sentence: str, shingle_size: int) -> set[str]:
    if shingle_size <= 0:
        raise ValueError("shingle_size must be positive")

    tokens = tokenize_for_minhash(sentence)
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        return {" ".join(tokens)}

    return {
        " ".join(tokens[index:index + shingle_size])
        for index in range(len(tokens) - shingle_size + 1)
    }


def stable_hash_int(value: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def minhash_signature(
    sentence: str,
    signature_size: int = MINHASH_SIGNATURE_SIZE,
    shingle_size: int = MINHASH_SHINGLE_SIZE,
) -> tuple[int, ...]:
    if signature_size <= 0:
        raise ValueError("signature_size must be positive")

    shingles = build_word_shingles(sentence, shingle_size=shingle_size)
    if not shingles:
        return tuple()

    return tuple(
        min(stable_hash_int(shingle, seed) for shingle in shingles)
        for seed in range(signature_size)
    )


def minhash_similarity(
    first_signature: tuple[int, ...],
    second_signature: tuple[int, ...],
) -> float:
    if not first_signature or not second_signature:
        return 0.0
    if len(first_signature) != len(second_signature):
        raise ValueError("MinHash signatures must have the same length")

    match_count = sum(
        first_value == second_value
        for first_value, second_value in zip(first_signature, second_signature)
    )
    return match_count / len(first_signature)


def deduplicate_near_duplicate_sentence_candidates(
    sentence_df: pd.DataFrame,
    similarity_threshold: float = MINHASH_DUPLICATE_THRESHOLD,
    signature_size: int = MINHASH_SIGNATURE_SIZE,
    shingle_size: int = MINHASH_SHINGLE_SIZE,
    band_size: int = MINHASH_LSH_BAND_SIZE,
) -> pd.DataFrame:
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if band_size <= 0:
        raise ValueError("band_size must be positive")
    if signature_size % band_size != 0:
        raise ValueError("signature_size must be divisible by band_size")
    if sentence_df.empty:
        result = sentence_df.copy()
        result["deduplication_method"] = "minhash"
        result["deduplication_threshold"] = similarity_threshold
        return result

    kept_indices = []
    kept_signatures = []
    seen_exact_sentences = set()
    band_buckets = {}

    for row_index, row in sentence_df.iterrows():
        sentence = row["sentence"]
        if not isinstance(sentence, str):
            continue

        normalized_sentence = " ".join(tokenize_for_minhash(sentence))
        if normalized_sentence in seen_exact_sentences:
            continue

        signature = minhash_signature(
            sentence,
            signature_size=signature_size,
            shingle_size=shingle_size,
        )
        candidate_signature_indices = set()
        for band_start in range(0, len(signature), band_size):
            band = signature[band_start:band_start + band_size]
            bucket_key = (band_start, band)
            candidate_signature_indices.update(band_buckets.get(bucket_key, []))

        is_duplicate = any(
            minhash_similarity(
                signature,
                kept_signatures[kept_signature_index],
            ) >= similarity_threshold
            for kept_signature_index in candidate_signature_indices
        )
        if is_duplicate:
            continue

        kept_indices.append(row_index)
        kept_signatures.append(signature)
        seen_exact_sentences.add(normalized_sentence)
        kept_signature_index = len(kept_signatures) - 1
        for band_start in range(0, len(signature), band_size):
            band = signature[band_start:band_start + band_size]
            bucket_key = (band_start, band)
            band_buckets.setdefault(bucket_key, []).append(kept_signature_index)

    result = sentence_df.loc[kept_indices].reset_index(drop=True)
    result["deduplication_method"] = "minhash"
    result["deduplication_threshold"] = similarity_threshold
    return result


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


def sentence_artifact_respects_sampling_limits(
    sentence_df: pd.DataFrame,
    sentences_per_polygon: int,
    sentences_per_url: int,
    target_polygon_count: int,
    sentence_filter_profile: str = SENTENCE_FILTER_PROFILE,
) -> bool:
    required_columns = [
        "osm_type",
        "osm_id",
        "url",
        "deduplication_method",
        "query_language",
        "sentence_filter_profile",
    ]
    if any(column not in sentence_df.columns for column in required_columns):
        return False
    if sentence_df.empty:
        return False
    if sentence_df["query_language"].isna().any():
        return False
    if not sentence_df["sentence_filter_profile"].eq(sentence_filter_profile).all():
        return False

    per_url_counts = sentence_df.groupby(
        ["osm_type", "osm_id", "url"],
        dropna=False,
    ).size()
    per_polygon_counts = sentence_df.groupby(
        ["osm_type", "osm_id"],
        dropna=False,
    ).size()

    return bool(
        per_url_counts.le(sentences_per_url).all()
        and per_polygon_counts.eq(sentences_per_polygon).all()
        and len(per_polygon_counts) == target_polygon_count
    )

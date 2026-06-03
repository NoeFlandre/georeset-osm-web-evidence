import re

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

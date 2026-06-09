import re

from georeset_osm_web_evidence.text.utils import count_words

TERMINAL_PUNCTUATION = (".", "!", "?", "。", "！", "؟")
TRAILING_CLOSERS = "\"'”’)]}"
ELLIPSIS_ENDINGS = ("...", "…")
MAX_SYMBOL_TO_WORD_RATIO = 0.1
SENTENCE_FILTER_PROFILE = "fineweb_inspired_v1"
SENTENCE_FILTER_RULES = (
    "8_to_80_words",
    "terminal_punctuation",
    "no_terminal_ellipsis",
    "symbol_to_word_ratio_lte_0.1",
)


def split_sentences(text: str) -> list[str]:
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    return re.split(r"(?<=[.!?])\s+", clean_text)


def has_terminal_punctuation(sentence: str) -> bool:
    stripped_sentence = sentence.strip().rstrip(TRAILING_CLOSERS)
    return stripped_sentence.endswith(TERMINAL_PUNCTUATION)


def ends_with_ellipsis(sentence: str) -> bool:
    stripped_sentence = sentence.strip().rstrip(TRAILING_CLOSERS)
    return stripped_sentence.endswith(ELLIPSIS_ENDINGS)


def symbol_to_word_ratio(sentence: str) -> float:
    word_count = count_words(sentence)
    if word_count == 0:
        return 0.0

    symbol_count = sentence.count("#") + sentence.count("…")
    symbol_count += sentence.count("...")
    return symbol_count / word_count


def is_sentence_candidate(
    sentence: str, min_word_count: int = 8, max_word_count: int = 80
) -> bool:
    word_count = count_words(sentence)
    if word_count < min_word_count:
        return False
    if word_count > max_word_count:
        return False
    if not has_terminal_punctuation(sentence):
        return False
    if ends_with_ellipsis(sentence):
        return False
    if symbol_to_word_ratio(sentence) > MAX_SYMBOL_TO_WORD_RATIO:
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

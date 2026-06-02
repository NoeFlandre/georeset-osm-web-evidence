import re


def split_non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def analyze_text_quality(text: str) -> dict:
    lines = split_non_empty_lines(text)

    line_count = len(lines)
    word_count = count_words(text)
    if line_count == 0:
        mean_words_per_line = 0

    else:
        mean_words_per_line = word_count / line_count

    return {
        "line_count": line_count,
        "word_count": word_count,
        "mean_words_per_line": mean_words_per_line,
        "duplicate_line_fraction": 0,
        "short_line_fraction": 0,
        "quality_flags": [],
    }

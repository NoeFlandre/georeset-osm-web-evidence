from collections import Counter

import pandas as pd

from georeset_osm_web_evidence.text.utils import count_words


def split_non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def count_duplicate_lines(lines: list[str]) -> int:
    counts = Counter(lines)
    duplicate_line_count = 0

    for line in lines:
        if counts[line] > 1:
            duplicate_line_count += 1

    return duplicate_line_count


def compute_quality_score(quality_flags: list[str]) -> float:
    if "empty_text" in quality_flags:
        return 0.0

    score = 1.0
    if "many_short_lines" in quality_flags:
        score -= 0.3

    if "duplicate_lines" in quality_flags:
        score -= 0.2

    return max(score, 0.0)


def analyze_text_quality(text: str) -> dict:
    lines = split_non_empty_lines(text)
    short_line_count = 0
    quality_flags = []

    line_count = len(lines)
    word_count = count_words(text)

    if line_count == 0:
        mean_words_per_line = 0
        short_line_fraction = 0
        duplicate_line_fraction = 0
        quality_flags.append("empty_text")

    else:
        mean_words_per_line = word_count / line_count
        duplicate_lines_count = count_duplicate_lines(lines)
        duplicate_line_fraction = duplicate_lines_count / line_count
        if duplicate_line_fraction != 0:
            quality_flags.append("duplicate_lines")

        for line in lines:
            words_per_line = count_words(line)
            if words_per_line <= 2:
                short_line_count += 1

        short_line_fraction = short_line_count / line_count
        if short_line_fraction >= 0.5:
            quality_flags.append("many_short_lines")

    quality_score = compute_quality_score(quality_flags)

    return {
        "line_count": line_count,
        "word_count": word_count,
        "mean_words_per_line": mean_words_per_line,
        "duplicate_line_fraction": duplicate_line_fraction,
        "short_line_fraction": short_line_fraction,
        "quality_flags": quality_flags,
        "quality_score": quality_score,
    }


def add_quality_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    quality_rows = result["text"].apply(
        lambda text: analyze_text_quality(text if isinstance(text, str) else "")
    )

    quality_df = pd.DataFrame(quality_rows.to_list())

    concat_df = pd.concat(
        [result.reset_index(drop=True), quality_df.reset_index(drop=True)],
        axis=1,
    )

    return concat_df

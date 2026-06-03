import re


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

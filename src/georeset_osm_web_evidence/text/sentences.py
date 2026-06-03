import re


def split_sentences(text: str) -> list[str]:
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    return re.split(r"(?<=[.!?])\s+", clean_text)

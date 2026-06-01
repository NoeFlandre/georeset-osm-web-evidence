import re
from html.parser import HTMLParser


class ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_tag_depth = 0
        self._inside_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_tag_depth += 1
        elif tag == "title":
            self._inside_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_tag_depth:
            self._ignored_tag_depth -= 1
        elif tag == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_tag_depth:
            return

        if self._inside_title:
            self.title_parts.append(data)
            return

        self.text_parts.append(data)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_readable_text(html: str) -> str:
    parser = ReadableTextParser()
    parser.feed(html)

    return normalize_whitespace(" ".join(parser.text_parts))


def extract_title(html: str) -> str | None:
    parser = ReadableTextParser()
    parser.feed(html)

    title = normalize_whitespace(" ".join(parser.title_parts))

    if not title:
        return None

    return title

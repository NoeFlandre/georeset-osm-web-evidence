from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import re

import requests


@dataclass
class PageTextResult:
    url: str
    final_url: str | None
    status_code: int | None
    title: str | None
    text: str | None
    text_length: int
    fetch_error: str | None


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


def fetch_page_text(
    url: str,
    timeout_seconds: int = 30,
    user_agent: str = "georeset-osm-web-evidence/0.1",
) -> dict:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
        )

        content_type = response.headers.get("content-type", "")

        if "text/html" not in content_type.lower():
            return asdict(
                PageTextResult(
                    url=url,
                    final_url=response.url,
                    status_code=response.status_code,
                    title=None,
                    text=None,
                    text_length=0,
                    fetch_error=f"Unsupported content type: {content_type}",
                )
            )

        response.raise_for_status()
        html = response.text
        text = extract_readable_text(html)

        if not text:
            return asdict(
                PageTextResult(
                    url=url,
                    final_url=response.url,
                    status_code=response.status_code,
                    title=extract_title(html),
                    text=None,
                    text_length=0,
                    fetch_error="No readable text extracted",
                )
            )

        return asdict(
            PageTextResult(
                url=url,
                final_url=response.url,
                status_code=response.status_code,
                title=extract_title(html),
                text=text,
                text_length=len(text),
                fetch_error=None,
            )
        )

    except requests.RequestException as error:
        return asdict(
            PageTextResult(
                url=url,
                final_url=None,
                status_code=None,
                title=None,
                text=None,
                text_length=0,
                fetch_error=str(error),
            )
        )

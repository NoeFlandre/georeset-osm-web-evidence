from dataclasses import dataclass

import trafilatura

from georeset_osm_web_evidence.web.html import extract_readable_text


@dataclass
class ExtractionResult:
    text: str | None
    method: str | None
    error: str | None


def extract_with_trafilatura(html: str, url: str | None = None) -> str | None:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
    )

    if text is None:
        return None

    text = text.strip()

    if text == "":
        return None

    return text


def extract_best_text(html: str, url: str | None = None) -> ExtractionResult:
    try:
        text = extract_with_trafilatura(html=html, url=url)

        if text is not None:
            return ExtractionResult(
                text=text,
                method="trafilatura",
                error=None,
            )

    except Exception as error:
        trafilatura_error = str(error)
    else:
        trafilatura_error = None

    fallback_text = extract_readable_text(html)

    if fallback_text:
        return ExtractionResult(
            text=fallback_text,
            method="html_parser",
            error=trafilatura_error,
        )

    return ExtractionResult(
        text=None, method=None, error=trafilatura_error or "No readable text extracted"
    )

from dataclasses import dataclass

import trafilatura


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
    except Exception as error:
        return ExtractionResult(
            text=None,
            method=None,
            error=str(error),
        )

    if text is not None:
        return ExtractionResult(
            text=text,
            method="trafilatura",
            error=None,
        )

    return ExtractionResult(
        text=None,
        method=None,
        error="No readable text extracted",
    )

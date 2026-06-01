from dataclasses import asdict, dataclass

import requests

from georeset_osm_web_evidence.web.extraction import extract_best_text
from georeset_osm_web_evidence.web.html import extract_title


@dataclass
class PageTextResult:
    url: str
    final_url: str | None
    status_code: int | None
    title: str | None
    text: str | None
    text_length: int
    fetch_error: str | None
    extraction_method: str | None
    extraction_error: str | None


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
                    extraction_method=None,
                    extraction_error=None,
                )
            )

        response.raise_for_status()
        html = response.text
        extraction = extract_best_text(html=html, url=response.url)
        text = extraction.text

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
                    extraction_method=extraction.method,
                    extraction_error=extraction.error,
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
                extraction_method=extraction.method,
                extraction_error=extraction.error,
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
                extraction_method=None,
                extraction_error=None,
            )
        )

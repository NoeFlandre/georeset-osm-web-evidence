import os
import time

import requests


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def normalize_brave_result(result: dict, query: str) -> dict:
    return {
        "provider": "brave",
        "query": query,
        "title": result.get("title"),
        "url": result.get("url"),
        "description": result.get("description"),
    }


def search_brave(
    query: str,
    count: int = 10,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_delay_seconds: int = 2,
    country: str = "FR",
    search_lang: str = "fr",
) -> list[dict]:
    api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")

    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY is not set")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                BRAVE_SEARCH_URL,
                params={
                    "q": query,
                    "count": count,
                    "country": country,
                    "search_lang": search_lang,
                },
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            raw_results = data.get("web", {}).get("results", [])

            return [
                normalize_brave_result(result, query=query)
                for result in raw_results
            ]

        except requests.RequestException as error:
            last_error = error

            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)

    raise last_error

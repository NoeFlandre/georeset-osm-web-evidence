import time

import requests


def geosearch_wikipedia(
    lat: float,
    lon: float,
    language: str,
    radius_m: int = 10_000,
    limit: int = 10,
    max_retries: int = 10,
    retry_delay_seconds: int = 10,
) -> list[dict]:
    url = f"https://{language}.wikipedia.org/w/api.php"

    last_error = None

    for attempt in range(1, max_retries + 1):
        response = requests.get(
            url,
            params={
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{lat}|{lon}",
                "gsradius": radius_m,
                "gslimit": limit,
                "format": "json",
            },
            headers={"User-Agent": "georeset_osm_web_evidence/0.1.0"},
            timeout=30,
        )

        try:
            response.raise_for_status()
            data = response.json()
            return data.get("query", {}).get("geosearch", [])
        except requests.RequestException as error:
            last_error = error
            print(f"Request to Wikipedia failed because of :{error}")
            if attempt < max_retries:
                delay = retry_delay_seconds * attempt
                print(f"Retrying in {delay} seconds")
                time.sleep(delay)
                continue

    raise last_error

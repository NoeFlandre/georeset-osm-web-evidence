import requests


def geosearch_wikipedia(
    lat: float,
    lon: float,
    language: str,
    radius_m: int = 10_000,
    limit: int = 10,
) -> list[dict]:
    url = f"https://{language}.wikipedia.org/w/api.php"

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

    response.raise_for_status()
    data = response.json()

    return data.get("query", {}).get("geosearch", [])

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_overpass_json(query: str) -> dict:
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=240,
    )
    response.raise_for_status()
    return response.json()

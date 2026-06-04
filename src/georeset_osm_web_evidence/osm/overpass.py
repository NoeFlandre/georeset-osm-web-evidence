import time

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def build_polygon_query(
    south: float,
    west: float,
    north: float,
    east: float,
    tags: list[tuple[str, str]],
    require_name: bool = False,
) -> str:
    name_filter = '["name"]' if require_name else ""
    tag_filters = "\n".join(
        f' way["{key}"="{value}"]{name_filter}({south},{west},{north},{east});\n'  # a way is a single closed polygon
        f' relation["{key}"="{value}"]{name_filter}({south},{west},{north},{east});'  # a relation is more complex polygon, often having holes or multiple parts
        for key, value in tags
    )
    # we want the results back within 240s as a json of the gemoetry and the tags
    return f"""
[out:json][timeout:240];
(
{tag_filters}
);
out geom;
    """


def fetch_overpass_json(
    query: str,
    max_retries: int = 3,
    retry_delay_seconds: int = 10,
) -> dict:
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Trying to request Overpass, attempt {attempt}/{max_retries}")
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={
                    "User-Agent": "georeset_osm_web_evidence/0.1.0",
                    "Accept": "application/json",
                },
                timeout=240,
            )

            if not response.ok:
                print(query)
                print(response.text)

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            last_error = error
            print(f"Overpass request failed on attempt {attempt}/{max_retries}")

        if attempt < max_retries:
            time.sleep(retry_delay_seconds)
            print(f"Sleeping for {retry_delay_seconds}")

    raise last_error

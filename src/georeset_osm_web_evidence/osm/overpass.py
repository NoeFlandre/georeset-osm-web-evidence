import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def build_polygon_query(
    south: float,
    west: float,
    north: float,
    east: float,
    tags: list[tuple[str, str]],
) -> str:
    tag_filters = "\n".join(
        f' way["{key}"="{value}"]({south},{west},{north},{east});\n'  # a way is a single closed polygon
        f' relation["{key}"="{value}"]({south},{west},{north},{east});'  # a relation is more complex polygon, often having holes or multiple parts
        for key, value in tags
    )
    # we want the results back within 180s as a json of the gemoetry and the tags
    return f"""
[out:json][timeout:180];
(
{tag_filters}
);
out geom;
    """


def fetch_overpass_json(query: str) -> dict:
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

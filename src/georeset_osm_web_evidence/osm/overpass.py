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
        f' way["{key}"="{value}"]({south}, {west}, {north}, {east});\n'  # a way is a single closed polygon
        f' relation["{key}"="{value}"]({south}, {west},{north},{east});'  # a relation is more complex polygon, often having holes or multiple parts
        for key, value in tags
    )
    # we want the results back within 180s as a json of the gemoetry and the tags
    return f"""
[out:json][timeout:180];
(
{tag_filters}
);
out geom tags;
    """


def fetch_overpass_json(query: str) -> dict:
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=240,
    )
    response.raise_for_status()
    return response.json()

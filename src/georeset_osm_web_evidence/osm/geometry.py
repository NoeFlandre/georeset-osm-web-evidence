from shapely.geometry import Polygon


def element_to_polygon(element: dict) -> Polygon | None:
    geometry = element.get("geometry")

    if not geometry:
        return None

    coordinates = [(point["lon"], point["lat"]) for point in geometry]

    if len(coordinates) < 4:
        return None

    return Polygon(coordinates)


def element_to_record(element: dict) -> dict | None:
    polygon = element_to_polygon(element)

    if not polygon:
        return None

    return {
        "osm_type": element.get("type"),
        "osm_id": element.get("id"),
        "osm_tags": element.get("tags", {}),
        "geometry": polygon,
    }

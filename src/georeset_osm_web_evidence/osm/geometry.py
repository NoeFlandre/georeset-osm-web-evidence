from shapely.geometry import Polygon


def element_to_polygon(element: dict) -> Polygon | None:
    geometry = element.get("geometry")

    if not geometry:
        return None

    coordinates = [(point["lon"], point["lat"]) for point in geometry]

    if len(coordinates) < 4:
        return None

    return Polygon(coordinates)

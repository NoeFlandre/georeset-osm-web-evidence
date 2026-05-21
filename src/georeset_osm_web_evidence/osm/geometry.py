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

    if polygon is None:
        return None

    return {
        "osm_type": element.get("type"),
        "osm_id": element.get("id"),
        "osm_tags": element.get("tags", {}),
        "geometry": polygon,
    }


def elements_to_records(elements: list[dict]) -> list[dict]:
    records = []

    for element in elements:
        record = element_to_record(element)

        if record is not None:
            records.append(record)

    return records

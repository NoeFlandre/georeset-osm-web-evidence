import math

from georeset_osm_web_evidence.osm.geodataframe import WGS84_GEOD


def geodesic_distance_km(
    lon_a: float,
    lat_a: float,
    lon_b: float,
    lat_b: float,
) -> float:
    _, _, distance_m = WGS84_GEOD.inv(lon_a, lat_a, lon_b, lat_b)

    return distance_m / 1_000


def is_far_enough_from_points(
    lon: float,
    lat: float,
    points: list[tuple[float, float]],
    min_distance_km: float,
) -> bool:
    return all(
        geodesic_distance_km(lon, lat, selected_lon, selected_lat)
        >= min_distance_km
        for selected_lon, selected_lat in points
    )


def distance_cell_size_degrees(min_distance_km: float) -> float:
    if min_distance_km <= 0:
        return 1.0

    return max(min_distance_km / 111, 0.25)


def distance_cell_key(
    lon: float,
    lat: float,
    cell_size_degrees: float,
) -> tuple[int, int]:
    return (
        math.floor(lat / cell_size_degrees),
        math.floor(lon / cell_size_degrees),
    )


def nearby_distance_cells(
    lon: float,
    lat: float,
    cell_size_degrees: float,
    min_distance_km: float,
) -> list[tuple[int, int]]:
    lat_cell, lon_cell = distance_cell_key(lon, lat, cell_size_degrees)
    lat_radius_degrees = min_distance_km / 110.574
    lon_scale = max(111.320 * math.cos(math.radians(lat)), 1)
    lon_radius_degrees = min_distance_km / lon_scale
    lat_steps = math.ceil(lat_radius_degrees / cell_size_degrees)
    lon_steps = math.ceil(lon_radius_degrees / cell_size_degrees)

    return [
        (nearby_lat_cell, nearby_lon_cell)
        for nearby_lat_cell in range(lat_cell - lat_steps, lat_cell + lat_steps + 1)
        for nearby_lon_cell in range(lon_cell - lon_steps, lon_cell + lon_steps + 1)
    ]


def add_point_to_distance_grid(
    grid: dict[tuple[int, int], list[tuple[float, float]]],
    lon: float,
    lat: float,
    cell_size_degrees: float,
) -> None:
    grid.setdefault(
        distance_cell_key(lon, lat, cell_size_degrees),
        [],
    ).append((lon, lat))


def is_far_enough_from_distance_grid(
    lon: float,
    lat: float,
    grid: dict[tuple[int, int], list[tuple[float, float]]],
    cell_size_degrees: float,
    min_distance_km: float,
) -> bool:
    for cell_key in nearby_distance_cells(
        lon,
        lat,
        cell_size_degrees,
        min_distance_km,
    ):
        if not is_far_enough_from_points(
            lon,
            lat,
            grid.get(cell_key, []),
            min_distance_km,
        ):
            return False

    return True

import unittest

from georeset_osm_web_evidence.osm.spatial_distance import (
    add_point_to_distance_grid,
    distance_cell_size_degrees,
    geodesic_distance_km,
    is_far_enough_from_distance_grid,
)


class SpatialDistanceTests(unittest.TestCase):
    def test_geodesic_distance_uses_wgs84_coordinates(self) -> None:
        distance_km = geodesic_distance_km(
            lon_a=0,
            lat_a=0,
            lon_b=1,
            lat_b=0,
        )

        self.assertGreater(distance_km, 111)
        self.assertLess(distance_km, 112)

    def test_distance_grid_rejects_points_inside_minimum_distance(self) -> None:
        cell_size = distance_cell_size_degrees(min_distance_km=100)
        grid = {}

        add_point_to_distance_grid(
            grid=grid,
            lon=0,
            lat=0,
            cell_size_degrees=cell_size,
        )

        self.assertFalse(
            is_far_enough_from_distance_grid(
                lon=0.2,
                lat=0,
                grid=grid,
                cell_size_degrees=cell_size,
                min_distance_km=100,
            )
        )
        self.assertTrue(
            is_far_enough_from_distance_grid(
                lon=2,
                lat=0,
                grid=grid,
                cell_size_degrees=cell_size,
                min_distance_km=100,
            )
        )


if __name__ == "__main__":
    unittest.main()

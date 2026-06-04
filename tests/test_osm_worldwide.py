import unittest

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_osm_web_evidence.osm.geodataframe import add_geodesic_area_km2
from georeset_osm_web_evidence.osm.overpass import build_polygon_query
from georeset_osm_web_evidence.osm.worldwide import (
    WORLDWIDE_TRAINING_BBOXES,
    add_area_size_bin,
    compute_sample_size,
    filter_named_environmental_polygons,
    generate_bbox_expansions,
    sample_worldwide_polygons,
)


class OsmWorldwideTests(unittest.TestCase):
    def test_build_polygon_query_can_require_named_elements(self) -> None:
        query = build_polygon_query(
            south=0,
            west=1,
            north=2,
            east=3,
            tags=[("landuse", "forest")],
            require_name=True,
        )

        self.assertIn('way["landuse"="forest"]["name"](0,1,2,3);', query)
        self.assertIn('relation["landuse"="forest"]["name"](0,1,2,3);', query)

    def test_geodesic_area_is_plausible_for_worldwide_polygons(self) -> None:
        gdf = gpd.GeoDataFrame(
            [{"geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])}],
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = add_geodesic_area_km2(gdf)

        self.assertGreater(result.loc[0, "area_km2"], 12_000)
        self.assertLess(result.loc[0, "area_km2"], 13_000)

    def test_area_size_bin_labels_small_medium_and_large_polygons(self) -> None:
        gdf = gpd.GeoDataFrame(
            [
                {"area_km2": 2},
                {"area_km2": 20},
                {"area_km2": 200},
                {"area_km2": 1_200},
            ]
        )

        result = add_area_size_bin(gdf)

        self.assertEqual(
            result["area_size_bin"].to_list(),
            ["small", "medium", "large", "very_large"],
        )

    def test_compute_sample_size_uses_planned_sentences_per_polygon(self) -> None:
        self.assertEqual(compute_sample_size(50_000, 10), 5_000)
        self.assertEqual(compute_sample_size(50_001, 10), 5_001)

    def test_generate_bbox_expansions_builds_neighboring_bbox_targets(self) -> None:
        anchor = {
            "bbox_id": "fr_test",
            "bbox_label": "France test",
            "country": "France",
            "world_region": "Europe",
            "local_language": "fr",
            "bbox": (47.8, -4.4, 48.2, -3.7),
        }

        expansions = generate_bbox_expansions([anchor], expansion_radius_steps=1)

        self.assertEqual(len(expansions), 8)
        self.assertEqual({bbox["country"] for bbox in expansions}, {"France"})
        self.assertTrue(
            all("_exp_" in bbox["bbox_id"] for bbox in expansions)
        )
        for bbox in expansions:
            south, west, north, east = bbox["bbox"]
            self.assertLessEqual(north - south, 0.8)
            self.assertLessEqual(east - west, 1.0)

    def test_generate_bbox_expansions_interleaves_anchor_targets(self) -> None:
        anchors = [
            {
                "bbox_id": "fr_test",
                "bbox_label": "France test",
                "country": "France",
                "world_region": "Europe",
                "local_language": "fr",
                "bbox": (47.8, -4.4, 48.2, -3.7),
            },
            {
                "bbox_id": "ke_test",
                "bbox_label": "Kenya test",
                "country": "Kenya",
                "world_region": "Africa",
                "local_language": "sw",
                "bbox": (-0.4, 36.7, 0.1, 37.4),
            },
        ]

        expansions = generate_bbox_expansions(anchors, expansion_radius_steps=1)

        self.assertTrue(expansions[0]["bbox_id"].startswith("fr_test"))
        self.assertTrue(expansions[1]["bbox_id"].startswith("ke_test"))

    def test_sample_worldwide_polygons_limits_each_bbox(self) -> None:
        rows = []
        for bbox_id in ["a", "b", "c"]:
            for osm_id in range(5):
                rows.append(
                    {
                        "bbox_id": bbox_id,
                        "osm_type": "way",
                        "osm_id": f"{bbox_id}-{osm_id}",
                        "geometry": Polygon(
                            [
                                (osm_id, 0),
                                (osm_id + 0.1, 0),
                                (osm_id + 0.1, 0.1),
                                (osm_id, 0.1),
                                (osm_id, 0),
                            ]
                        ),
                    }
                )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=6,
            max_per_bbox=2,
            random_state=7,
        )

        self.assertEqual(len(sample), 6)
        self.assertTrue((sample["bbox_id"].value_counts() <= 2).all())
        self.assertEqual(set(sample["bbox_id"]), {"a", "b", "c"})

    def test_sample_worldwide_polygons_limits_each_country(self) -> None:
        rows = []
        for country in ["France", "Spain", "Italy"]:
            for bbox_index in range(2):
                for polygon_index in range(4):
                    lon = bbox_index * 10 + polygon_index
                    rows.append(
                        {
                            "bbox_id": f"{country}-{bbox_index}",
                            "country": country,
                            "world_region": "Europe",
                            "area_size_bin": "small",
                            "osm_type": "way",
                            "osm_id": f"{country}-{bbox_index}-{polygon_index}",
                            "geometry": Polygon(
                                [
                                    (lon, 0),
                                    (lon + 0.1, 0),
                                    (lon + 0.1, 0.1),
                                    (lon, 0.1),
                                    (lon, 0),
                                ]
                            ),
                        }
                    )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=9,
            max_per_bbox=4,
            max_per_country=3,
            random_state=7,
        )

        self.assertEqual(len(sample), 9)
        self.assertTrue((sample["country"].value_counts() <= 3).all())
        self.assertEqual(set(sample["country"]), {"France", "Spain", "Italy"})

    def test_sample_worldwide_polygons_balances_regions_when_downsampling(self) -> None:
        rows = []
        for region in ["Africa", "Asia", "Europe"]:
            for bbox_index in range(3):
                for polygon_index in range(5):
                    rows.append(
                        {
                            "bbox_id": f"{region}-{bbox_index}",
                            "world_region": region,
                            "osm_type": "way",
                            "osm_id": f"{region}-{bbox_index}-{polygon_index}",
                            "geometry": Polygon(
                                [
                                    (polygon_index, 0),
                                    (polygon_index + 0.1, 0),
                                    (polygon_index + 0.1, 0.1),
                                    (polygon_index, 0.1),
                                    (polygon_index, 0),
                                ]
                            ),
                        }
                    )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=9,
            max_per_bbox=5,
            random_state=7,
        )

        self.assertEqual(len(sample), 9)
        self.assertEqual(sample["world_region"].value_counts().to_dict(), {
            "Africa": 3,
            "Asia": 3,
            "Europe": 3,
        })

    def test_sample_worldwide_polygons_balances_area_bins_when_available(self) -> None:
        rows = []
        for area_bin in ["small", "medium", "large"]:
            for polygon_index in range(5):
                rows.append(
                    {
                        "bbox_id": f"bbox-{area_bin}-{polygon_index}",
                        "world_region": "Africa",
                        "area_size_bin": area_bin,
                        "osm_type": "way",
                        "osm_id": f"{area_bin}-{polygon_index}",
                        "geometry": Polygon(
                            [
                                (polygon_index, 0),
                                (polygon_index + 0.1, 0),
                                (polygon_index + 0.1, 0.1),
                                (polygon_index, 0.1),
                                (polygon_index, 0),
                            ]
                        ),
                    }
                )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=6,
            max_per_bbox=5,
            random_state=7,
        )

        self.assertEqual(len(sample), 6)
        self.assertEqual(sample["area_size_bin"].value_counts().to_dict(), {
            "large": 2,
            "medium": 2,
            "small": 2,
        })

    def test_filter_named_environmental_polygons_keeps_only_named_matching_tags(self) -> None:
        gdf = gpd.GeoDataFrame(
            [
                {
                    "osm_tags": {"name": "Named forest", "landuse": "forest"},
                    "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                },
                {
                    "osm_tags": {"name": "Named road", "highway": "track"},
                    "geometry": Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
                },
                {
                    "osm_tags": {"landuse": "forest"},
                    "geometry": Polygon([(4, 0), (5, 0), (5, 1), (4, 0)]),
                },
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = filter_named_environmental_polygons(gdf)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].osm_tags["name"], "Named forest")

    def test_sample_worldwide_polygons_prefers_sparse_centroids(self) -> None:
        rows = []
        for index, lon in enumerate([0, 0.01, 0.02, 2, 4, 6]):
            rows.append(
                {
                    "bbox_id": "same-bbox",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                    "osm_type": "way",
                    "osm_id": index,
                    "geometry": Polygon(
                        [
                            (lon, 0),
                            (lon + 0.1, 0),
                            (lon + 0.1, 0.1),
                            (lon, 0.1),
                            (lon, 0),
                        ]
                    ),
                }
            )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=4,
            max_per_bbox=6,
            min_centroid_distance_km=100,
            random_state=7,
        )

        self.assertEqual(len(sample), 4)
        self.assertLessEqual(len(set(sample["osm_id"]) & {0, 1, 2}), 1)

    def test_sample_worldwide_polygons_prefers_global_sparse_centroids(self) -> None:
        rows = []
        coordinates = [
            ("bbox-a", 0),
            ("bbox-b", 0.01),
            ("bbox-c", 0.02),
            ("bbox-d", 3),
            ("bbox-e", 6),
            ("bbox-f", 9),
        ]
        for index, (bbox_id, lon) in enumerate(coordinates):
            rows.append(
                {
                    "bbox_id": bbox_id,
                    "country": "France",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                    "osm_type": "way",
                    "osm_id": index,
                    "geometry": Polygon(
                        [
                            (lon, 0),
                            (lon + 0.1, 0),
                            (lon + 0.1, 0.1),
                            (lon, 0.1),
                            (lon, 0),
                        ]
                    ),
                }
            )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=4,
            max_per_bbox=1,
            min_global_centroid_distance_km=100,
            random_state=7,
        )

        self.assertEqual(len(sample), 4)
        self.assertLessEqual(len(set(sample["osm_id"]) & {0, 1, 2}), 1)

    def test_sample_worldwide_polygons_keeps_global_sparsity_under_target(self) -> None:
        rows = []
        coordinates = [
            ("bbox-a", 0),
            ("bbox-b", 0.01),
            ("bbox-c", 0.02),
            ("bbox-d", 3),
        ]
        for index, (bbox_id, lon) in enumerate(coordinates):
            rows.append(
                {
                    "bbox_id": bbox_id,
                    "world_region": "Europe",
                    "area_size_bin": "small",
                    "osm_type": "way",
                    "osm_id": index,
                    "geometry": Polygon(
                        [
                            (lon, 0),
                            (lon + 0.1, 0),
                            (lon + 0.1, 0.1),
                            (lon, 0.1),
                            (lon, 0),
                        ]
                    ),
                }
            )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=10,
            max_per_bbox=1,
            min_global_centroid_distance_km=100,
            random_state=7,
        )

        self.assertLess(len(sample), len(gdf))
        self.assertLessEqual(len(set(sample["osm_id"]) & {0, 1, 2}), 1)

    def test_sample_worldwide_polygons_fills_shortfall_after_sparse_choices(self) -> None:
        rows = []
        for index, lon in enumerate([0, 0.01, 0.02, 2, 4]):
            rows.append(
                {
                    "bbox_id": "same-bbox",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                    "osm_type": "way",
                    "osm_id": index,
                    "geometry": Polygon(
                        [
                            (lon, 0),
                            (lon + 0.1, 0),
                            (lon + 0.1, 0.1),
                            (lon, 0.1),
                            (lon, 0),
                        ]
                    ),
                }
            )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        sample = sample_worldwide_polygons(
            gdf,
            sample_size=5,
            max_per_bbox=5,
            min_centroid_distance_km=100,
            random_state=7,
        )

        self.assertEqual(len(sample), 5)

    def test_training_bboxes_are_unique_and_cover_major_world_regions(self) -> None:
        bbox_ids = [bbox["bbox_id"] for bbox in WORLDWIDE_TRAINING_BBOXES]
        regions = {bbox["world_region"] for bbox in WORLDWIDE_TRAINING_BBOXES}

        self.assertEqual(len(bbox_ids), len(set(bbox_ids)))
        self.assertEqual(
            regions,
            {
                "Africa",
                "Asia",
                "Europe",
                "North America",
                "Oceania",
                "South America",
            },
        )

    def test_training_bboxes_include_russia(self) -> None:
        countries = {bbox["country"] for bbox in WORLDWIDE_TRAINING_BBOXES}

        self.assertIn("Russia", countries)

    def test_training_bboxes_are_small_enough_for_overpass(self) -> None:
        for bbox in WORLDWIDE_TRAINING_BBOXES:
            south, west, north, east = bbox["bbox"]
            self.assertLessEqual(north - south, 0.8, bbox["bbox_id"])
            self.assertLessEqual(east - west, 1.0, bbox["bbox_id"])


if __name__ == "__main__":
    unittest.main()

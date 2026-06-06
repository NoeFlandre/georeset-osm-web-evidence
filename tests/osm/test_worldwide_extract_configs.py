import unittest

from georeset_osm_web_evidence.osm.worldwide_extract_configs import (
    DEFAULT_LANGUAGE_BY_REGION,
    EXTRACT_CONFIGS,
    REGION_BY_GEOFABRIK_ROOT,
    SKIP_DISCOVERED_EXTRACT_IDS,
    configured_world_regions,
)


class WorldwideExtractConfigTests(unittest.TestCase):
    def test_configured_world_regions_are_sorted_from_geofabrik_roots(self) -> None:
        self.assertEqual(
            configured_world_regions(),
            [
                "Africa",
                "Asia",
                "Europe",
                "North America",
                "Oceania",
                "South America",
            ],
        )

    def test_extract_configs_have_required_metadata_and_unique_ids(self) -> None:
        extract_ids = [config["extract_id"] for config in EXTRACT_CONFIGS]

        self.assertEqual(len(extract_ids), len(set(extract_ids)))
        for config in EXTRACT_CONFIGS:
            self.assertIsInstance(config["extract_id"], str)
            self.assertIn(config["world_region"], configured_world_regions())
            self.assertIn("local_language", config)

    def test_discovered_extract_defaults_cover_all_regions(self) -> None:
        self.assertEqual(
            set(DEFAULT_LANGUAGE_BY_REGION),
            set(configured_world_regions()),
        )
        self.assertTrue(set(REGION_BY_GEOFABRIK_ROOT).issubset(
            SKIP_DISCOVERED_EXTRACT_IDS
        ))


if __name__ == "__main__":
    unittest.main()

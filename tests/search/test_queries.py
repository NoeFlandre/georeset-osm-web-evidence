import unittest

from georeset_osm_web_evidence.osm.worldwide_bboxes import WORLDWIDE_TRAINING_BBOXES
from georeset_osm_web_evidence.osm.worldwide_extract_configs import (
    DEFAULT_LANGUAGE_BY_REGION,
    EXTRACT_CONFIGS,
)
from georeset_osm_web_evidence.search.queries import (
    build_contextual_english_search_queries,
    build_location_topic_english_search_queries,
    build_search_queries,
)
from georeset_osm_web_evidence.search.terms import TERMS_BY_LANGUAGE


class SearchQueryTests(unittest.TestCase):
    def test_builds_french_queries_by_default(self):
        queries = build_search_queries({"name": "Forêt Alpha", "landuse": "forest"})

        self.assertEqual(
            queries,
            [
                '"Forêt Alpha" forêt',
                '"Forêt Alpha" biodiversité',
                '"Forêt Alpha" gestion forestière',
                '"Forêt Alpha" environnement',
            ],
        )

    def test_builds_queries_for_explicit_search_languages(self):
        queries = build_search_queries(
            {"name": "Forêt Alpha", "landuse": "forest"},
            search_languages=["fr", "en"],
        )

        self.assertEqual(
            queries,
            [
                '"Forêt Alpha" forêt',
                '"Forêt Alpha" biodiversité',
                '"Forêt Alpha" gestion forestière',
                '"Forêt Alpha" environnement',
                '"Forêt Alpha" forest',
                '"Forêt Alpha" biodiversity',
                '"Forêt Alpha" forest management',
                '"Forêt Alpha" environment',
            ],
        )

    def test_rejects_unsupported_search_language(self):
        with self.assertRaises(ValueError):
            build_search_queries(
                {"name": "Forêt Alpha", "landuse": "forest"},
                search_languages=["xx"],
            )

    def test_removes_duplicate_queries_across_languages(self):
        queries = build_search_queries(
            {"name": "Marais Alpha", "natural": "wetland"},
            search_languages=["fr", "en"],
        )

        self.assertEqual(len(queries), len(set(queries)))
        self.assertEqual(queries.count('"Marais Alpha" conservation'), 1)

    def test_builds_spanish_local_language_queries(self):
        queries = build_search_queries(
            {
                "name": "Parque-Reserva Natural de las Quinientas",
                "leisure": "nature_reserve",
            },
            search_languages=["es"],
        )

        self.assertIn(
            '"Parque-Reserva Natural de las Quinientas" reserva natural',
            queries,
        )
        self.assertIn(
            '"Parque-Reserva Natural de las Quinientas" conservación',
            queries,
        )

    def test_builds_sinhala_local_language_queries(self):
        queries = build_search_queries(
            {"name": "Rice", "landuse": "farmland"},
            search_languages=["si"],
        )

        self.assertIn('"Rice" කෘෂිකර්මය', queries)
        self.assertIn('"Rice" වගාව', queries)

    def test_terms_cover_all_configured_worldwide_local_languages(self):
        configured_languages = (
            {config["local_language"] for config in EXTRACT_CONFIGS}
            | {bbox["local_language"] for bbox in WORLDWIDE_TRAINING_BBOXES}
            | set(DEFAULT_LANGUAGE_BY_REGION.values())
        )

        self.assertEqual(
            sorted(configured_languages - set(TERMS_BY_LANGUAGE)),
            [],
        )

    def test_builds_contextual_english_queries_with_location_and_category(self):
        queries = build_contextual_english_search_queries(
            osm_tags={"name": "Sagole Baobab", "leisure": "nature_reserve"},
            country="South Africa",
            world_region="Africa",
            source_extract_id="south-africa",
            polygon_category="protected_area",
            max_queries=4,
        )

        self.assertEqual(len(queries), 4)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertEqual(
            queries,
            [
                '"Sagole Baobab" "South Africa" nature reserve',
                '"Sagole Baobab" "Africa" nature reserve',
                '"Sagole Baobab" "South Africa" protection',
                '"Sagole Baobab" "South Africa" protected area',
            ],
        )

    def test_contextual_english_queries_expand_us_extract_context(self):
        queries = build_contextual_english_search_queries(
            osm_tags={"name": "Marion Meadows", "natural": "wetland"},
            country="us/idaho",
            world_region="North America",
            source_extract_id="us/idaho",
            polygon_category="wetland",
            max_queries=4,
        )

        self.assertIn('"Marion Meadows" "Idaho" wetland', queries)
        self.assertIn('"Marion Meadows" "United States" wetland', queries)
        self.assertTrue(all("Marion Meadows" in query for query in queries))

    def test_builds_location_topic_queries_with_one_template(self):
        queries = build_location_topic_english_search_queries(
            osm_tags={"name": "Sagole Baobab", "leisure": "nature_reserve"},
            country="South Africa",
            world_region="Africa",
            source_extract_id="south-africa",
            polygon_category="protected_area",
            max_queries=4,
        )

        self.assertEqual(
            queries,
            [
                '"Sagole Baobab" "South Africa" "nature reserve"',
                '"Sagole Baobab" "South Africa" "protection"',
                '"Sagole Baobab" "South Africa" "biodiversity"',
                '"Sagole Baobab" "South Africa" "conservation"',
            ],
        )

    def test_location_topic_queries_use_best_available_region_context(self):
        queries = build_location_topic_english_search_queries(
            osm_tags={"name": "Marion Meadows", "natural": "wetland"},
            country="us/idaho",
            world_region="North America",
            source_extract_id="us/idaho",
            polygon_category="wetland",
            max_queries=2,
        )

        self.assertEqual(
            queries,
            [
                '"Marion Meadows" "Idaho" "wetland"',
                '"Marion Meadows" "Idaho" "biodiversity"',
            ],
        )


if __name__ == "__main__":
    unittest.main()

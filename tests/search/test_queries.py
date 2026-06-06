import unittest

from georeset_osm_web_evidence.osm.worldwide_bboxes import WORLDWIDE_TRAINING_BBOXES
from georeset_osm_web_evidence.osm.worldwide_extract_configs import (
    DEFAULT_LANGUAGE_BY_REGION,
    EXTRACT_CONFIGS,
)
from georeset_osm_web_evidence.search.queries import build_search_queries
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


if __name__ == "__main__":
    unittest.main()

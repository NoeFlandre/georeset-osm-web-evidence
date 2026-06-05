import unittest

from georeset_osm_web_evidence.search.queries import build_search_queries


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


if __name__ == "__main__":
    unittest.main()

import unittest

from shapely.geometry import Polygon

from georeset_osm_web_evidence.wikipedia.spatial import (
    article_is_inside_polygon,
    filter_articles_inside_polygon,
)


class WikipediaSpatialTests(unittest.TestCase):
    def test_article_inside_polygon_uses_covering_boundary_semantics(self):
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])

        self.assertTrue(article_is_inside_polygon({"lon": 0.5, "lat": 0.25}, polygon))
        self.assertTrue(article_is_inside_polygon({"lon": 0.0, "lat": 0.0}, polygon))
        self.assertFalse(article_is_inside_polygon({"lon": 2.0, "lat": 2.0}, polygon))

    def test_filter_articles_inside_polygon_keeps_only_covered_articles(self):
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        articles = [
            {"title": "Inside", "lon": 0.5, "lat": 0.25},
            {"title": "Outside", "lon": 2.0, "lat": 2.0},
        ]

        result = filter_articles_inside_polygon(articles, polygon)

        self.assertEqual(result, [articles[0]])


if __name__ == "__main__":
    unittest.main()

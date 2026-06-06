import unittest
from unittest.mock import patch

from georeset_osm_web_evidence.web.extraction import extract_best_text


class WebExtractionTests(unittest.TestCase):
    def test_extracts_main_article_text(self):
        html = """
        <html>
            <body>
                <nav>Home Contact Cookie policy</nav>
                <article>
                    <h1>Forêt test</h1>
                    <p>This protected forest contains wetlands and walking trails.</p>
                </article>
                <footer>Legal links</footer>
            </body>
        </html>
        """

        result = extract_best_text(html, url="https://example.test/forest")

        self.assertIn("protected forest", result.text)
        self.assertIsNone(result.error)
        self.assertEqual(result.method, "trafilatura")

    def test_returns_no_text_when_trafilatura_extracts_nothing(self):
        html = """
        <html>
            <body>
                <p>Short text about wetlands and forest trails.</p>
            </body>
        </html>
        """

        with patch(
            "georeset_osm_web_evidence.web.extraction.extract_with_trafilatura",
            return_value=None,
        ):
            result = extract_best_text(html)

        self.assertIsNone(result.text)
        self.assertIsNone(result.method)
        self.assertEqual(result.error, "No readable text extracted")

    def test_reports_trafilatura_error_without_alternate_extractor(self):
        with patch(
            "georeset_osm_web_evidence.web.extraction.extract_with_trafilatura",
            side_effect=RuntimeError("parser failed"),
        ):
            result = extract_best_text("<html><body>text</body></html>")

        self.assertIsNone(result.text)
        self.assertIsNone(result.method)
        self.assertEqual(result.error, "parser failed")


if __name__ == "__main__":
    unittest.main()

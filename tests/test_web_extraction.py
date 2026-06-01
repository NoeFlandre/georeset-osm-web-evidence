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
        self.assertIn(result.method, {"trafilatura", "html_parser"})

    def test_falls_back_to_html_parser_when_trafilatura_returns_none(self):
        html = """
        <html>
            <body>
                <p>Fallback text about wetlands and forest trails.</p>
            </body>
        </html>
        """

        with patch(
            "georeset_osm_web_evidence.web.extraction.extract_with_trafilatura",
            return_value=None,
        ):
            result = extract_best_text(html)

        self.assertIn("Fallback text", result.text)
        self.assertEqual(result.method, "html_parser")
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()

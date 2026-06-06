import unittest
from unittest.mock import patch

from georeset_osm_web_evidence.web.html import extract_title
from georeset_osm_web_evidence.web.text import fetch_page_text


class FakeResponse:
    url = "https://example.test/page"
    status_code = 200
    headers = {"content-type": "text/html; charset=utf-8"}
    text = "<html><body><script>onlyJavascript()</script></body></html>"

    def raise_for_status(self):
        return None


class FakeReadableText:
    url = "https://example.test/readable-page"
    status_code = 200
    headers = {"content-type": "text/html; charset=utf-8"}
    text = """
    <html>
        <body>
            <article>
                <p>Forest evidence text about wetlands and trails.</p>
            </article>
        </body>
    </html>
    """

    def raise_for_status(self):
        return None


class WebTextTests(unittest.TestCase):
    def test_extracts_html_title(self):
        html = """
        <html>
          <head>
            <title>
                Forest management plan
            </title>
            <style>.hidden { display: none; }</style>
            <script>console.log("ignore me")</script>
          </head>
          <body>
            <h1>Forest management plan</h1>
            <p>The forest is protected.</p>
            <p>It contains wetlands and trails.</p>
          </body>
        </html>
        """

        title = extract_title(html)

        self.assertEqual(title, "Forest management plan")

    def test_fetch_marks_empty_extracted_text_as_error(self):
        with patch("georeset_osm_web_evidence.web.text.requests.get") as get:
            get.return_value = FakeResponse()

            result = fetch_page_text("https://example.test/page")

        self.assertEqual(result["text_length"], 0)
        self.assertEqual(result["fetch_error"], "No readable text extracted")

    def test_fetch_page_text_reports_extraction_method(self):
        with patch("georeset_osm_web_evidence.web.text.requests.get") as get:
            get.return_value = FakeReadableText()
            result = fetch_page_text("https://example.test/readable-page")

        self.assertIn("Forest evidence text", result["text"])
        self.assertIsNone(result["fetch_error"])
        self.assertEqual(result["extraction_method"], "trafilatura")
        self.assertIsNone(result["extraction_error"])


if __name__ == "__main__":
    unittest.main()

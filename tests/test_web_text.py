import unittest
from unittest.mock import patch

from georeset_osm_web_evidence.web.text import extract_readable_text, fetch_page_text


class FakeResponse:
    url = "https://example.test/page"
    status_code = 200
    headers = {"content-type": "text/html; charset=utf-8"}
    text = "<html><body><script>onlyJavascript()</script></body></html>"

    def raise_for_status(self):
        return None


class ExtractReadableTextTests(unittest.TestCase):
    def test_extracts_visible_text_and_ignores_script_and_style(self):
        html = """
        <html>
          <head>
            <title>Ignored title</title>
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

        text = extract_readable_text(html)

        self.assertIn("Forest management plan", text)
        self.assertIn("The forest is protected.", text)
        self.assertIn("It contains wetlands and trails.", text)
        self.assertNotIn("console.log", text)
        self.assertNotIn("display: none", text)

    def test_fetch_marks_empty_extracted_text_as_error(self):
        with patch("georeset_osm_web_evidence.web.text.requests.get") as get:
            get.return_value = FakeResponse()

            result = fetch_page_text("https://example.test/page")

        self.assertEqual(result["text_length"], 0)
        self.assertEqual(result["fetch_error"], "No readable text extracted")


if __name__ == "__main__":
    unittest.main()

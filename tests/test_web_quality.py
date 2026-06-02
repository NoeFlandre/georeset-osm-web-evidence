import unittest

from georeset_osm_web_evidence.web.quality import analyze_text_quality


class WebQualityTests(unittest.TestCase):
    def test_analyze_basic_text_shape(self):
        text = """
            forest path wetland
            river bird habitat
            local site protected
            """

        result = analyze_text_quality(text)

        self.assertEqual(result["line_count"], 3)
        self.assertEqual(result["word_count"], 9)
        self.assertEqual(result["mean_words_per_line"], 3)
        self.assertEqual(result["duplicate_line_fraction"], 0)
        self.assertEqual(result["short_line_fraction"], 0)
        self.assertEqual(result["quality_flags"], [])

    def test_detects_many_short_lines(self):
        text = """
        Pizza
        Pesto
        Pasta
        This forest is actually cool
        """

        result = analyze_text_quality(text)

        self.assertEqual(result["word_count"], 8)
        self.assertEqual(result["line_count"], 4)
        self.assertEqual(result["short_line_fraction"], 3 / 4)
        self.assertIn("many_short_lines", result["quality_flags"])

    def test_handles_empty_text(self):
        text = ""

        result = analyze_text_quality(text)

        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["line_count"], 0)
        self.assertEqual(result["mean_words_per_line"], 0)
        self.assertEqual(result["short_line_fraction"], 0)
        self.assertEqual(result["duplicate_line_fraction"], 0)
        self.assertIn("empty_text", result["quality_flags"])


if __name__ == "__main__":
    unittest.main()

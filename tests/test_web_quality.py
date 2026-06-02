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


if __name__ == "__main__":
    unittest.main()

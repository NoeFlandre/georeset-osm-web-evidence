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

    def test_detects_duplicate_lines(self):
        text = """
        This is an example of line
        This is an example of line
        This is once again another line
        """

        result = analyze_text_quality(text)

        self.assertEqual(result["word_count"], 18)
        self.assertEqual(result["line_count"], 3)
        self.assertEqual(result["mean_words_per_line"], 6)
        self.assertEqual(result["short_line_fraction"], 0)
        self.assertEqual(result["duplicate_line_fraction"], 2 / 3)
        self.assertIn("duplicate_lines", result["quality_flags"])

    def test_assigns_quality_score(self):
        clean_text = """
        The anatomy of the kākāpō typifies the tendency of bird evolution on oceanic islands.
        With few predators and abundant food, kākāpō exhibit island syndrome development, having a generally robust torso physique at the expense of flight abilities, resulting in reduced shoulder and wing muscles, along with a diminished keel on the sternum.
        """

        empty_text = ""

        clean_result = analyze_text_quality(clean_text)
        empty_result = analyze_text_quality(empty_text)

        self.assertEqual(clean_result["quality_score"], 1.0)
        self.assertEqual(empty_result["quality_score"], 0.0)

    def test_quality_flags_should_reduce_quality_score(self):
        empty_text = """"""

        duplicate_lines_text = """
        This is a duplicate line
        This is a duplicate line
        """

        many_short_lines_text = """
        Hello
        My
        Name
        Is
        Jack
        """

        empty_result = analyze_text_quality(empty_text)
        duplicate_result = analyze_text_quality(duplicate_lines_text)
        many_short_lines_result = analyze_text_quality(many_short_lines_text)

        self.assertEqual(empty_result["quality_score"], 0.0)
        self.assertEqual(duplicate_result["quality_score"], 0.8)
        self.assertEqual(many_short_lines_result["quality_score"], 0.7)


if __name__ == "__main__":
    unittest.main()

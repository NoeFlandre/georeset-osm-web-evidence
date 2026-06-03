import unittest

from georeset_osm_web_evidence.text.sentences import split_sentences


class TestSentences(unittest.TestCase):
    def test_split_simple_sentences(self):
        text = """
        This is an example of text. I am trying to figure out whether I can split sentences. Shouldn't be too complicated.
        """

        sentences = split_sentences(text)

        self.assertEqual(
            sentences,
            [
                "This is an example of text.",
                "I am trying to figure out whether I can split sentences.",
                "Shouldn't be too complicated.",
            ],
        )

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("  \n  "), [])


if __name__ == "__main__":
    unittest.main()

import unittest

from georeset_osm_web_evidence.text.sentences import (
    extract_sentence_candidates,
    is_sentence_candidate,
    split_sentences,
)


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

    def test_filters_sentence_based_on_word_count(self):
        self.assertFalse(is_sentence_candidate("Too short"))
        self.assertTrue(
            is_sentence_candidate(
                "This sentence is a valid candidate because it has enough words."
            )
        )

    def test_sentence_candidate_word_count_boundaries_are_inclusive(self):
        eight_words = "one two three four five six seven eight"
        eighty_words = " ".join(f"word{i}" for i in range(80))
        eighty_one_words = " ".join(f"word{i}" for i in range(81))

        self.assertTrue(is_sentence_candidate(eight_words))
        self.assertTrue(is_sentence_candidate(eighty_words))
        self.assertFalse(is_sentence_candidate(eighty_one_words))

    def test_extracts_candidate_sentences(self):
        text = """
        Home.
        This is a sentence which I would like to keep because it is long enough.
        Menu.
        This is another sentence that I would like to be kept since it is valuable.
        """
        sentence_candidates = extract_sentence_candidates(text)

        self.assertEqual(
            sentence_candidates,
            [
                "This is a sentence which I would like to keep because it is long enough.",
                "This is another sentence that I would like to be kept since it is valuable.",
            ],
        )


if __name__ == "__main__":
    unittest.main()

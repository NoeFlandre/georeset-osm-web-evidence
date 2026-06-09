import unittest

import pandas as pd

from scripts.evidence.build_english_only_sentence_pilot import (
    filter_english_candidate_urls,
)


class EnglishSentencePilotScriptTests(unittest.TestCase):
    def test_filter_english_candidate_urls_keeps_only_english_query_rows(self):
        candidate_urls_df = pd.DataFrame(
            [
                {"url": "https://example.org/en", "query_language": "en"},
                {"url": "https://example.org/fr", "query_language": "fr"},
            ]
        )

        result = filter_english_candidate_urls(candidate_urls_df)

        self.assertEqual(result["url"].to_list(), ["https://example.org/en"])


if __name__ == "__main__":
    unittest.main()

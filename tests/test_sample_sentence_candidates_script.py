import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.evidence.sample_sentence_candidates import run_sentence_candidate_sampling


class TestSampleSentenceCandidatesScript(unittest.TestCase):
    def test_samples_sentence_candidates_and_writes_output(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "polygon_name": "Forêt Alpha",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/a",
                    "quality_score": 1.0,
                    "sentence": "This is a strong candidate sentence about forest management.",
                },
                {
                    "polygon_name": "Forêt Alpha",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/a",
                    "quality_score": 0.9,
                    "sentence": "This is another strong candidate sentence about biodiversity.",
                },
                {
                    "polygon_name": "Marais Beta",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/b",
                    "quality_score": 0.5,
                    "sentence": "This low quality sentence should not be sampled.",
                },
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "sentence_candidates.parquet"
            output_path = Path(temporary_directory) / "sampled_sentences.parquet"
            sentence_df.to_parquet(input_path, index=False)

            sampled_df = run_sentence_candidate_sampling(
                input_path=input_path,
                output_path=output_path,
                sample_size=2,
                min_quality_score=0.8,
                random_state=42,
            )

            saved_df = pd.read_parquet(output_path)

        self.assertTrue((saved_df["quality_score"] >= 0.8).all())
        self.assertEqual(len(sampled_df), 2)
        self.assertEqual(len(saved_df), 2)
        self.assertEqual(sampled_df["sentence"].to_list(), saved_df["sentence"].to_list())


if __name__ == "__main__":
    unittest.main()

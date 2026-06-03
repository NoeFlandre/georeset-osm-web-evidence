import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.sample_sentence_candidates import (
    sample_sentence_candidates,
)


class TestSampleSentenceCandidates(unittest.TestCase):
    def test_sample_sentence_candidates(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forêt Alpha",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/a",
                    "quality_score": 1.0,
                    "sentence": "This is a strong candidate sentence about forest management and biodiversity.",
                },
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Forêt Alpha",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/a",
                    "quality_score": 0.8,
                    "sentence": "This sentence is also useful because it contains enough local context.",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Marais Beta",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/b",
                    "quality_score": 1.0,
                    "sentence": "This wetland sentence describes species, habitat, and environmental protection.",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Marais Beta",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/b",
                    "quality_score": 0.5,
                    "sentence": "This low quality sentence should not survive the quality filter.",
                },
                {
                    "osm_type": "relation",
                    "osm_id": 3,
                    "polygon_name": "Réserve Gamma",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/c",
                    "quality_score": 0.9,
                    "sentence": "This protected area sentence has useful information for future labeling.",
                },
            ]
        )

        sampled_sentence_df = sample_sentence_candidates(
            sentence_df,
            sample_size=3,
            min_quality_score=0.8,
            random_state=42,
        )

        sampled_sentence_df_2 = sample_sentence_candidates(
            sentence_df,
            sample_size=3,
            min_quality_score=0.8,
            random_state=42,
        )

        self.assertEqual(len(sampled_sentence_df), 3)
        self.assertTrue((sampled_sentence_df["quality_score"] >= 0.8).all())
        self.assertEqual(
            sampled_sentence_df["sentence"].to_list(),
            sampled_sentence_df_2["sentence"].to_list(),
        )


if __name__ == "__main__":
    unittest.main()

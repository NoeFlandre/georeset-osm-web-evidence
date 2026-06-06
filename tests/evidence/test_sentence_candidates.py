import unittest

import pandas as pd

from georeset_osm_web_evidence.evidence.sentence_candidates import (
    SENTENCE_CANDIDATE_COLUMNS,
    build_sentence_candidate_dataframe,
    filter_english_sentence_candidates,
    limit_sentence_candidates,
    looks_like_english_sentence,
    select_complete_sentence_candidates,
)


class TestEvidenceSentenceCandidate(unittest.TestCase):
    def test_builds_one_row_per_sentence_candidate(self):
        text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 123,
                    "polygon_name": "Forêt Test",
                    "has_wikipedia_articles": True,
                    "url": "https://example.com/page",
                    "final_url": "https://example.com/final",
                    "search_title": "Search title",
                    "search_description": "Search description",
                    "search_queries": '"Forêt Test" biodiversité',
                    "title": "Page title",
                    "text_length": 200,
                    "quality_score": 1.0,
                    "quality_flags": [],
                    "text": (
                        "Home. "
                        "This sentence contains enough words to become a useful candidate. "
                        "Map. "
                        "Another sentence contains enough useful words for labeling."
                    ),
                },
                {
                    "osm_type": "way",
                    "osm_id": 456,
                    "polygon_name": "Empty Text",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/empty",
                    "final_url": "https://example.com/empty",
                    "search_title": "Empty",
                    "search_description": "Empty",
                    "search_queries": '"Empty" biodiversité',
                    "title": "Empty title",
                    "text_length": 0,
                    "quality_score": 0.0,
                    "quality_flags": ["empty_text"],
                    "text": None,
                },
            ]
        )

        sentence_df = build_sentence_candidate_dataframe(text_df)

        self.assertEqual(len(sentence_df), 2)
        self.assertEqual(
            sentence_df["polygon_name"].to_list(), ["Forêt Test", "Forêt Test"]
        )
        self.assertEqual(sentence_df["osm_id"].to_list(), [123, 123])
        self.assertEqual(
            sentence_df["sentence"].to_list(),
            [
                "This sentence contains enough words to become a useful candidate.",
                "Another sentence contains enough useful words for labeling.",
            ],
        )
        self.assertIn("quality_score", sentence_df.columns)
        self.assertIn("search_queries", sentence_df.columns)

    def test_returns_expected_schema_when_no_sentence_survives(self):
        text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 456,
                    "polygon_name": "Empty Text",
                    "has_wikipedia_articles": False,
                    "url": "https://example.com/empty",
                    "final_url": "https://example.com/empty",
                    "search_title": "Empty",
                    "search_description": "Empty",
                    "search_queries": '"Empty" biodiversité',
                    "title": "Empty title",
                    "text_length": 0,
                    "quality_score": 0.0,
                    "quality_flags": ["empty_text"],
                    "text": None,
                }
            ]
        )

        sentence_df = build_sentence_candidate_dataframe(text_df)

        self.assertTrue(sentence_df.empty)
        self.assertEqual(sentence_df.columns.to_list(), SENTENCE_CANDIDATE_COLUMNS)

    def test_limits_sentence_candidates_per_url_then_per_polygon(self):
        sentence_rows = []
        for url_index in range(12):
            for sentence_index in range(2):
                sentence_rows.append(
                    {
                        "osm_type": "way",
                        "osm_id": 123,
                        "polygon_name": "Forêt Test",
                        "url": f"https://example.com/page-{url_index}",
                        "sentence": f"Sentence {url_index}-{sentence_index}",
                    }
                )
        for url_index in range(3):
            for sentence_index in range(2):
                sentence_rows.append(
                    {
                        "osm_type": "relation",
                        "osm_id": 456,
                        "polygon_name": "Wetland Test",
                        "url": f"https://example.org/page-{url_index}",
                        "sentence": f"Wetland sentence {url_index}-{sentence_index}",
                    }
                )
        sentence_df = pd.DataFrame(sentence_rows)

        limited_df = limit_sentence_candidates(
            sentence_df,
            max_sentences_per_polygon=10,
            max_sentences_per_url=1,
        )

        first_polygon_df = limited_df[limited_df["osm_id"] == 123]
        second_polygon_df = limited_df[limited_df["osm_id"] == 456]

        self.assertEqual(len(first_polygon_df), 10)
        self.assertEqual(len(second_polygon_df), 3)
        self.assertTrue(
            limited_df.groupby(["osm_type", "osm_id", "url"]).size().le(1).all()
        )
        self.assertEqual(
            first_polygon_df["sentence"].to_list(),
            [f"Sentence {url_index}-0" for url_index in range(10)],
        )

    def test_selects_only_polygons_with_complete_sentence_quota(self):
        sentence_rows = []
        for url_index in range(12):
            sentence_rows.append(
                {
                    "osm_type": "way",
                    "osm_id": 123,
                    "polygon_name": "Complete Forest",
                    "url": f"https://complete.example/page-{url_index}",
                    "sentence": f"Complete sentence {url_index}",
                }
            )
        for url_index in range(9):
            sentence_rows.append(
                {
                    "osm_type": "way",
                    "osm_id": 456,
                    "polygon_name": "Incomplete Forest",
                    "url": f"https://incomplete.example/page-{url_index}",
                    "sentence": f"Incomplete sentence {url_index}",
                }
            )
        sentence_df = pd.DataFrame(sentence_rows)

        complete_df = select_complete_sentence_candidates(
            sentence_df,
            sentences_per_polygon=10,
            sentences_per_url=1,
        )

        self.assertEqual(len(complete_df), 10)
        self.assertEqual(complete_df["polygon_name"].unique().tolist(), ["Complete Forest"])
        self.assertEqual(
            complete_df["sentence"].to_list(),
            [f"Complete sentence {url_index}" for url_index in range(10)],
        )

    def test_filters_english_sentence_candidates(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "query_local_language": "en",
                    "sentence": "The forest contains wetlands and open grassland habitats.",
                },
                {
                    "query_local_language": "en",
                    "sentence": "Reserva Extrativista de São João da Ponta Área protegida.",
                },
                {
                    "query_local_language": "fr",
                    "sentence": "The forest contains wetlands and open grassland habitats.",
                },
            ]
        )

        english_df = filter_english_sentence_candidates(sentence_df)

        self.assertEqual(
            english_df["sentence"].to_list(),
            ["The forest contains wetlands and open grassland habitats."],
        )

    def test_english_sentence_heuristic_rejects_dutch_false_positive(self):
        self.assertFalse(
            looks_like_english_sentence(
                "De plas is daarna in beheer van Staatsbosbeheer gekomen."
            )
        )
        self.assertTrue(
            looks_like_english_sentence(
                "The lake is now managed by the national forestry agency."
            )
        )


if __name__ == "__main__":
    unittest.main()

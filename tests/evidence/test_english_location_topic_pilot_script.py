import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.evidence import build_english_location_topic_sentence_pilot as pilot


class EnglishLocationTopicPilotScriptTests(unittest.TestCase):
    def test_experiment_paths_are_isolated(self):
        self.assertEqual(
            pilot.OUTPUT_DIR,
            Path(
                "data/processed/pilots/"
                "worldwide_sentence_pilot_10_english_location_topic_queries_v1"
            ),
        )
        self.assertNotEqual(pilot.OUTPUT_DIR, pilot.ENGLISH_ONLY_OUTPUT_DIR)

    def test_builds_location_topic_search_artifacts_with_injected_search(self):
        pilot_gdf = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "osm_tags": {
                        "name": "Sagole Baobab",
                        "leisure": "nature_reserve",
                    },
                    "polygon_name": "Sagole Baobab",
                    "has_wikipedia_articles": False,
                    "country": "South Africa",
                    "world_region": "Africa",
                    "source_extract_id": "south-africa",
                    "polygon_category": "protected_area",
                    "local_language": "en",
                    "query_local_language": "en",
                    "area_size_bin": "tiny",
                }
            ]
        )
        seen_queries = []

        def fake_search(query: str, count: int, **kwargs):
            seen_queries.append(query)
            self.assertEqual(kwargs["country"], "US")
            self.assertEqual(kwargs["search_lang"], "en")
            return [
                {
                    "provider": "brave",
                    "title": "Search result",
                    "url": f"https://example.org/{len(seen_queries)}",
                    "description": "Description",
                }
            ]

        search_results_df, search_attempts_df = pilot.build_location_topic_search_artifacts(
            pilot_gdf,
            search_func=fake_search,
            sleep_func=lambda _seconds: None,
            results_per_query=1,
            request_delay_seconds=0,
        )

        self.assertEqual(
            seen_queries,
            [
                '"Sagole Baobab" "South Africa" "nature reserve"',
                '"Sagole Baobab" "South Africa" "protection"',
                '"Sagole Baobab" "South Africa" "biodiversity"',
                '"Sagole Baobab" "South Africa" "conservation"',
            ],
        )
        self.assertEqual(search_results_df["query_language"].unique().tolist(), ["en"])
        self.assertEqual(search_attempts_df["query_language"].unique().tolist(), ["en"])

    def test_writes_location_aware_prompt_artifacts(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": "https://example.org/a",
                    "sentence": "The reserve contains extensive wetland habitat.",
                    "polygon_name": "Sagole Baobab",
                    "country": "South Africa",
                    "world_region": "Africa",
                    "polygon_category": "protected_area",
                    "title": "Sagole Baobab travel guide",
                    "search_queries": '"Sagole Baobab" "South Africa" "protected area"',
                }
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "sentence_candidates.parquet"
            parquet_output_path = temp_path / "llm_requests.parquet"
            jsonl_output_path = temp_path / "llm_requests.jsonl"
            sentence_df.to_parquet(input_path, index=False)

            prompt_df = pilot.run_location_topic_labeling_request_build(
                input_path=input_path,
                parquet_output_path=parquet_output_path,
                jsonl_output_path=jsonl_output_path,
            )

            saved_df = pd.read_parquet(parquet_output_path)
            jsonl_exists = jsonl_output_path.exists()

        self.assertEqual(len(prompt_df), 1)
        self.assertEqual(len(saved_df), 1)
        self.assertTrue(jsonl_exists)
        self.assertIn("specific target polygon", saved_df.loc[0, "prompt"])

    def test_selects_exact_url_artifacts_from_sentence_urls(self):
        sentence_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": f"https://example.org/{index}",
                    "sentence": f"Sentence {index} contains enough words for testing.",
                }
                for index in range(10)
            ]
        )
        candidate_urls_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "url": f"https://example.org/{index}",
                    "best_rank": index,
                }
                for index in range(12)
            ]
        )
        page_text_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "source_url": f"https://example.org/{index}",
                    "text": "Page text",
                }
                for index in range(12)
            ]
        )

        selected_urls_df, selected_page_text_df = pilot.select_exact_url_artifacts(
            sentence_df=sentence_df,
            candidate_urls_df=candidate_urls_df,
            page_text_df=page_text_df,
            urls_per_polygon=10,
        )

        self.assertEqual(len(selected_urls_df), 10)
        self.assertEqual(len(selected_page_text_df), 10)
        self.assertEqual(selected_urls_df["url"].to_list(), sentence_df["url"].to_list())

    def test_orders_completion_candidates_by_underrepresented_region_and_area_bin(self):
        source_df = pd.DataFrame(
            [
                {
                    "osm_type": "way",
                    "osm_id": 1,
                    "polygon_name": "Already Complete",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                },
                {
                    "osm_type": "way",
                    "osm_id": 2,
                    "polygon_name": "Another Europe",
                    "world_region": "Europe",
                    "area_size_bin": "small",
                },
                {
                    "osm_type": "way",
                    "osm_id": 3,
                    "polygon_name": "Africa Medium Reserve",
                    "world_region": "Africa",
                    "area_size_bin": "medium",
                },
                {
                    "osm_type": "way",
                    "osm_id": 4,
                    "polygon_name": "Asia Large Forest",
                    "world_region": "Asia",
                    "area_size_bin": "large",
                },
            ]
        )
        complete_df = source_df.head(1)
        attempted_df = pd.DataFrame([{"osm_type": "way", "osm_id": 2}])

        candidates = pilot.order_completion_candidates(
            source_df=source_df,
            complete_df=complete_df,
            attempted_df=attempted_df,
        )

        self.assertEqual(candidates["osm_id"].to_list(), [3, 4])


if __name__ == "__main__":
    unittest.main()

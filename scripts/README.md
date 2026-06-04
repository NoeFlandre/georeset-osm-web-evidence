# Scripts

Small command-line entry points for running the pipeline stages. Run scripts from
the repository root with `uv run python scripts/<stage>/<script>.py`.

## OSM

- `osm/fetch_osm_polygons.py` fetches named environmental OSM polygons from configured France bounding boxes, filters them by area, adds centroids, deduplicates them, and writes `data/raw/osm/named_polygon_candidates.parquet`.
- `osm/sample_balanced_wikipedia_polygons.py` samples a 50/50 balanced dataset of polygons with and without Wikipedia articles.
- `osm/visualize_osm_samples.py` builds a Folium HTML map for the balanced sample and colors polygons by Wikipedia coverage.

## Wikipedia

- `wikipedia/check_wikipedia_geosearch_batch.py` enriches a batch of candidate polygons with French and English Wikipedia geosearch results, keeping only articles whose coordinates fall inside the polygon.

## Search

- `search/build_search_queries.py` prints tag-aware, name-based web search queries for the balanced sample, currently in French and English.
- `search/search_brave_sample.py` runs a small Brave Search smoke test on a few generated French/English queries. Requires `BRAVE_SEARCH_API_KEY`.
- `search/report_brave_search_coverage.py` reports which balanced polygons already have Brave Search results or logged Brave attempts, using the configured French/English query set.
- `search/collect_brave_search_results.py` appends Brave Search results for a balanced set of unsearched polygons, logs attempted polygons, and saves normalized result rows to `data/processed/search/brave_results_sample.parquet`. It currently uses French and English queries. Requires `BRAVE_SEARCH_API_KEY`.
- `search/prepare_search_result_urls.py` deduplicates Brave Search results into candidate evidence URLs and removes Wikipedia URLs.

## Evidence

- `evidence/fetch_candidate_page_text.py` fetches candidate URLs, extracts readable page text, and saves `data/processed/evidence/page_text_sample.parquet`.
- `evidence/add_quality_metadata.py` computes page-level text quality metadata and writes `data/processed/evidence/page_text_sample_with_quality_metadata.parquet`.
- `evidence/summarize_polygon_evidence.py` summarizes candidate URL, fetch, and high-quality evidence counts per polygon.
- `evidence/extract_sentence_candidate.py` converts fetched page text into sentence-level candidate rows.
- `evidence/sample_sentence_candidates.py` samples high-quality sentence candidates for manual inspection or future LLM labeling.
- `evidence/build_labeling_candidates.py` deduplicates high-quality sentence candidates, assigns stable IDs, and writes parquet plus JSONL inputs for future LLM labeling.

## Labeling

- `labeling/build_labeling_prompt_sample.py` prepares a small parquet and JSONL batch of binary `relevant`/`irrelevant` prompts from existing sentence-level labeling candidates. It does not call an LLM.
- `labeling/run_llama_cpp_labeling_sample.py` labels a tiny prompt batch with the configured Qwen GGUF through `llama-cpp-python`. This is intended for a remote GPU machine and requires `llama-cpp-python` plus the model there.

## Review

- `review/build_human_review_sample.py` converts successful fetched page text into a reviewer-friendly, capped-per-polygon CSV with empty human label and notes columns.

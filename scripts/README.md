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

- `search/build_search_queries.py` prints tag-aware, name-based web search queries for the balanced sample.
- `search/search_brave_sample.py` runs a small Brave Search smoke test on a few generated queries. Requires `BRAVE_SEARCH_API_KEY`.
- `search/collect_brave_search_results.py` collects Brave Search results for a limited polygon subset and saves normalized result rows to `data/processed/search/brave_results_sample.parquet`. Requires `BRAVE_SEARCH_API_KEY`.
- `search/prepare_search_result_urls.py` deduplicates Brave Search results into candidate evidence URLs and removes Wikipedia URLs.

## Evidence

- `evidence/fetch_candidate_page_text.py` fetches a small batch of candidate URLs, extracts readable page text, and saves `data/processed/evidence/page_text_sample.parquet`.

## Review

- `review/build_human_review_sample.py` converts fetched page text into a reviewer-friendly CSV with empty human label and notes columns.

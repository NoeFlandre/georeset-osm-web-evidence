# Scripts

Small command-line entry points for running the pipeline stages. Run scripts from the repository root with `uv run python scripts/<script>.py`.

## Pipeline scripts

- `fetch_osm_polygons.py` fetches named environmental OSM polygons from configured France bounding boxes, filters them by area, adds centroids, deduplicates them, and writes `data/raw/osm/named_polygon_candidates.parquet`.
- `check_wikipedia_geosearch_batch.py` enriches a batch of candidate polygons with French and English Wikipedia geosearch results, keeping only articles whose coordinates fall inside the polygon.
- `sample_balanced_wikipedia_polygons.py` samples a 50/50 balanced dataset of polygons with and without Wikipedia articles.
- `visualize_osm_samples.py` builds a Folium HTML map for the balanced sample and colors polygons by Wikipedia coverage.
- `build_search_queries.py` prints tag-aware, name-based web search queries for the balanced sample.
- `search_brave_sample.py` runs a small Brave Search smoke test on a few generated queries. Requires `BRAVE_SEARCH_API_KEY`.
- `collect_brave_search_results.py` collects Brave Search results for a limited polygon subset and saves normalized result rows to `data/processed/search/brave_results_sample.parquet`. Requires `BRAVE_SEARCH_API_KEY`.

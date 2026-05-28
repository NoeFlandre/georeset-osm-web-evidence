# Scripts

Small command-line entry points for running the pipeline stages. Run scripts from the repository root with `uv run python scripts/<script>.py`.

## Pipeline scripts

- `fetch_osm_polygons.py` fetches named environmental OSM polygons from configured France bounding boxes, filters them by area, adds centroids, deduplicates them, and writes `data/raw/osm_polygons_sample.parquet`.
- `check_wikipedia_geosearch_batch.py` enriches a batch of candidate polygons with French and English Wikipedia geosearch results, keeping only articles whose coordinates fall inside the polygon.
- `sample_balanced_wikipedia_polygons.py` samples a 50/50 balanced dataset of polygons with and without Wikipedia articles.
- `visualize_osm_samples.py` builds a Folium HTML map for the balanced sample and colors polygons by Wikipedia coverage.
- `build_search_queries.py` prints tag-aware, name-based web search queries for the balanced sample.
- `search_brave_sample.py` runs a small Brave Search smoke test on a few generated queries. Requires `BRAVE_SEARCH_API_KEY`.

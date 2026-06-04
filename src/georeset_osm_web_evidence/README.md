# georeset_osm_web_evidence

Core Python package for the OSM web-evidence pipeline.

The package is organized by responsibility:

- `osm/` fetches and prepares OSM polygon candidates.
- `wikipedia/` checks whether geolocated Wikipedia articles fall inside polygons.
- `search/` builds web search queries and calls search providers.
- `web/` fetches candidate evidence pages and extracts readable text.
- `labeling/` prepares binary sentence-labeling prompts and validates future LLM label outputs.
- `review/` prepares human-review tables from candidate evidence.
- `storage/` reads and writes local geospatial artifacts.
- `viz/` creates lightweight visual QA maps.

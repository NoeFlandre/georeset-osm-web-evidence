
# Pipeline overview

![Pipeline overview](docs/figures/project_overview.png)

Additional repository diagrams are available in [docs/diagrams.md](docs/diagrams.md).

This project builds a pipeline for collecting web evidence linked to named
environmental and agricultural OpenStreetMap (OSM) polygons. The codebase can:

- fetch and sample named OSM polygons from Overpass or Geofabrik extracts;
- enrich polygons with coordinate-based Wikipedia geosearch results;
- build tag-aware Brave Search queries in English and local languages;
- fetch candidate web pages and extract text with `trafilatura`;
- compute page- and sentence-level quality metadata;
- prepare human-review workbooks and LLM labeling inputs.

Open research and data-quality questions:

- Wikipedia URL handling: the search URL stage removes Wikipedia URLs because
  geotagged Wikipedia articles are already handled through the coordinate API.
  This can miss useful articles without coordinates or broader articles that
  mention a polygon, so the rule should be revisited before large-scale runs.
- Local-language metadata: worldwide search uses English plus a
  `query_local_language` value. Some Geofabrik-derived polygons still inherit
  broad regional defaults, so the full 5,000-polygon sample needs a curated
  extract/country-to-language validation pass before scaling.
- Spatial validation: evidence scoring can use named-entity linking,
  gazetteers, Wikipedia coordinates, embeddings, or satellite-derived features
  to compare fetched text with the target polygon.

We separate concerns by keeping the code in this repository while keeping the data in a hugging face bucket :

```
hf://buckets/NoeFlandre/georeset-osm-web-evidence
```

# OSM

Utilities for building the polygon candidate set from OpenStreetMap.

- `bboxes.py` defines French bounding boxes queried through Overpass.
- `tags.py` defines the environmental/agricultural OSM tags we fetch.
- `overpass.py` builds Overpass queries and fetches JSON with retry logic.
- `geometry.py` converts OSM elements into polygon records and keeps named records.
- `geodataframe.py` builds GeoDataFrames, computes France-specific projected areas or worldwide geodesic areas, filters by area, and adds centroids.
- `sampling.py` contains simple polygon sampling helpers.
- `spatial_distance.py` contains geodesic distance and distance-grid helpers for sparse polygon sampling.
- `worldwide.py` validates named environmental polygons, adds bbox metadata, creates log-scale area-size bins (`tiny`, `small`, `medium`, `large`), computes training sample sizes, and samples polygons with global sparsity plus per-cell, per-country, world-region, and area-size balancing.
- `worldwide_bboxes.py` contains the worldwide training bbox catalogue and bbox expansion helper.
- `worldwide_balancing.py` computes balanced per-group sample targets used by the worldwide sampler.
- `worldwide_extract_configs.py` contains curated Geofabrik extract configs, region defaults, and skipped root extract IDs for worldwide data collection.
- `worldwide_planning.py` contains pure planning helpers for region deficits, extract prioritization, and choosing the best balanced worldwide sample.

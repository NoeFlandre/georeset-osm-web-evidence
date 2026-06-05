# OSM

Utilities for building the polygon candidate set from OpenStreetMap.

- `bboxes.py` defines French bounding boxes queried through Overpass.
- `tags.py` defines the environmental/agricultural OSM tags we fetch.
- `overpass.py` builds Overpass queries and fetches JSON with retry logic.
- `geometry.py` converts OSM elements into polygon records and keeps named records.
- `geodataframe.py` builds GeoDataFrames, computes France-specific projected areas or worldwide geodesic areas, filters by area, and adds centroids.
- `sampling.py` contains simple polygon sampling helpers.
- `worldwide.py` defines worldwide training bounding boxes, validates named environmental polygons, adds bbox metadata, creates log-scale area-size bins (`tiny`, `small`, `medium`, `large`), computes training sample sizes, and samples polygons with global sparsity plus per-cell, per-country, world-region, and area-size balancing.

# OSM Tests

Tests for OpenStreetMap polygon ingestion and worldwide sampling.

- `test_extracts.py` checks Geofabrik multipolygon conversion and global spatial-cell assignment.
- `test_spatial_distance.py` checks geodesic distance and distance-grid behavior used for sparse sampling.
- `test_worldwide_bboxes.py` checks the worldwide bbox catalogue and expansion helper.
- `test_worldwide.py` checks named environmental filtering, log-scale area bins, bbox expansion, and sparse balanced worldwide sampling.
- `test_worldwide_balancing.py` checks pure per-group target allocation for balanced worldwide samples.
- `test_worldwide_extract_configs.py` checks curated Geofabrik extract config invariants.
- `test_worldwide_planning.py` checks pure planning helpers for extract prioritization and balanced sample selection.

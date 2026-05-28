# Wikipedia

Utilities for checking direct geolocated Wikipedia coverage.

- `api.py` queries French or English Wikipedia geosearch around polygon centroids.
- `spatial.py` keeps only Wikipedia articles whose geotagged point falls inside the polygon geometry.

Wikipedia geosearch is used as a coverage check, not as the final web-evidence source.

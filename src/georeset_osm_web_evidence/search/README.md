# Search

Utilities for producing and executing web search queries.

- `terms.py` stores category-specific query terms.
- `queries.py` extracts polygon names, classifies polygons from OSM tags, and builds tag-aware queries.
- `providers.py` calls external search providers and normalizes results into a provider-independent schema.

Current provider support is focused on Brave Search.

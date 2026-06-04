# Search

Utilities for producing and executing web search queries.

- `terms.py` stores category-specific query terms.
- `queries.py` extracts polygon names, classifies polygons from OSM tags, and builds tag-aware queries.
- `providers.py` calls external search providers and normalizes results into a provider-independent schema.

Current provider support is focused on Brave Search.

Query generation accepts an explicit `search_languages` list. The default is French
for backward compatibility, while the France search scripts currently request both
French and English with `["fr", "en"]`. This is an explicit configuration, not an
automatic local-language inference system.

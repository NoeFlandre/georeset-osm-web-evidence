# Search

Utilities for producing and executing web search queries.

- `terms.py` stores category-specific query terms.
- `queries.py` extracts polygon names, classifies polygons from OSM tags, and builds tag-aware queries.
- `languages.py` resolves the local query language used for worldwide search runs.
- `providers.py` calls external search providers and normalizes results into a provider-independent schema.

Current provider support is focused on Brave Search.

Query generation accepts an explicit `search_languages` list. Production search
scripts pass the intended languages directly, for example French and English with
`["fr", "en"]`. This is explicit configuration, not automatic local-language
inference.

For the worldwide sentence pilot, queries are built in English plus a resolved
local language. This resolved value is stored separately as `query_local_language`
so that the original polygon metadata remains traceable. The current resolver
handles the pilot cases that were known to inherit overly broad defaults, but it
is not yet a complete worldwide language normalization system. Before scaling the
pipeline beyond the pilot, the full 5,000-polygon sample should be checked against
a curated extract/country-to-language mapping.

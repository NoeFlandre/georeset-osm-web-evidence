# Web

This package fetches and normalizes web pages selected as possible evidence for
OSM polygons.

It should stay focused on generic web-page handling:

- HTTP fetching.
- Main-text extraction through `trafilatura`.
- Fetch metadata such as status codes and errors.

It should not decide which URLs are relevant to a polygon. That belongs to later
evidence review and evaluation code.

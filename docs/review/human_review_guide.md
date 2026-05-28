# Human Review Guide

Human review files are generated under `data/review/`.

Each row is one candidate web page for one OSM polygon. The reviewer should
decide whether the page is useful evidence for that polygon.

The review sample only includes pages whose text was fetched successfully.
Broken pages are kept in the upstream evidence artifact, but are not presented
for human relevance review.

## Labels

Use one of these values in `human_label`:

- `relevant`: the page is about the polygon itself or contains useful evidence
  about its location, environment, land use, protection status, history, or
  management.
- `irrelevant`: the page is about another place or only mentions the polygon in
  a weak or unhelpful way.
- `broken`: the page could not be fetched, has empty extracted text, or is not
  usable for review.
- `unclear`: the reviewer cannot decide quickly.

Use `human_notes` for short comments only when useful.

## Review Advice

Read `polygon_name`, `search_title`, `search_description`, `page_title`, and
`text_preview`.

Open `source_url` only when the preview is not enough.

Do not spend more than about one minute on one row. Use `unclear` when the row
needs more work than that.

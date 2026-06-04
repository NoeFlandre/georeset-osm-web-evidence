def combine_queries_for_review(queries) -> str:
    if queries is None:
        return ""

    return "; ".join(str(query) for query in queries)


def build_page_text_row(candidate_url_row, page_text: dict) -> dict:
    return {
        "osm_type": candidate_url_row.osm_type,
        "osm_id": candidate_url_row.osm_id,
        "polygon_name": candidate_url_row.polygon_name,
        "has_wikipedia_articles": candidate_url_row.has_wikipedia_articles,
        "provider": candidate_url_row.provider,
        "source_url": candidate_url_row.url,
        "search_title": candidate_url_row.title,
        "search_description": candidate_url_row.description,
        "search_queries": combine_queries_for_review(candidate_url_row.queries),
        **page_text,
    }

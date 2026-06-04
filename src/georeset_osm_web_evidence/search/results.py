from datetime import datetime, timezone


def result_to_row(
    polygon_row,
    polygon_name: str,
    query: str,
    rank: int,
    result: dict,
) -> dict:
    return {
        "osm_type": polygon_row.osm_type,
        "osm_id": polygon_row.osm_id,
        "polygon_name": polygon_name,
        "has_wikipedia_articles": polygon_row.has_wikipedia_articles,
        "query": query,
        "provider": result["provider"],
        "rank": rank,
        "title": result["title"],
        "url": result["url"],
        "description": result["description"],
    }


def attempt_to_row(
    polygon_row,
    polygon_name: str,
    query: str,
    result_count: int,
) -> dict:
    return {
        "osm_type": polygon_row.osm_type,
        "osm_id": polygon_row.osm_id,
        "polygon_name": polygon_name,
        "has_wikipedia_articles": polygon_row.has_wikipedia_articles,
        "query": query,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "result_count": result_count,
    }

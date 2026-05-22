from shapely.geometry import Point


def article_is_inside_polygon(article: dict, polygon) -> bool:
    point = Point(article["lon"], article["lat"])
    return polygon.covers(point)


def filter_articles_inside_polygon(articles: list[dict], polygon) -> list[dict]:
    return [
        article for article in articles if article_is_inside_polygon(article, polygon)
    ]

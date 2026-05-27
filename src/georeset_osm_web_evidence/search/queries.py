from georeset_osm_web_evidence.search.terms import (
    AGRICULTURE_TERMS,
    DEFAULT_TERMS,
    FOREST_TERMS,
    PROTECTED_AREA_TERMS,
    WETLAND_TERMS,
)


def clean_tags(osm_tags: dict) -> dict:
    return {key: value for key, value in osm_tags.items() if value is not None}


def get_osm_name(osm_tags: dict) -> str:
    tags = clean_tags(osm_tags)

    for key in ["name", "name:fr", "official_name", "short_name", "alt_name"]:
        name = tags.get(key)

        if isinstance(name, str) and name.strip() != "":
            return name.strip()

    raise ValueError(f"OSM tags do not contain a usable name")


def classify_polygon(osm_tags: dict) -> str:
    tags = clean_tags(osm_tags)

    landuse = tags.get("landuse")
    natural = tags.get("natural")
    leisure = tags.get("leisure")
    boundary = tags.get("boundary")

    if landuse == "forest" or natural == "wood":
        return "forest"

    if landuse in {"farmland", "meadow", "orchard", "vineyard"}:
        return "agriculture"

    if natural == "wetland" or "wetland" in tags:
        return "wetland"

    if leisure == "nature_reserve" or boundary == "protected_area":
        return "protected_area"

    return "default"


def get_query_terms(category: str) -> list[str]:
    if category == "forest":
        return FOREST_TERMS
    if category == "agriculture":
        return AGRICULTURE_TERMS
    if category == "wetland":
        return WETLAND_TERMS
    if category == "protected_area":
        return PROTECTED_AREA_TERMS
    return DEFAULT_TERMS


def build_search_queries(osm_tags: dict) -> list[str]:
    name = get_osm_name(osm_tags)
    category = classify_polygon(osm_tags)
    terms = get_query_terms(category)

    return [f'"{name}" {term}' for term in terms]

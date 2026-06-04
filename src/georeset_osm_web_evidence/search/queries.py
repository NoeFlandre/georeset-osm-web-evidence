from georeset_osm_web_evidence.search.terms import TERMS_BY_LANGUAGE


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


def get_query_terms(category: str, language: str = "fr") -> list[str]:
    if language not in TERMS_BY_LANGUAGE:
        supported_languages = ", ".join(sorted(TERMS_BY_LANGUAGE))
        raise ValueError(
            f"Unsupported search language '{language}'. "
            f"Supported languages: {supported_languages}"
        )

    terms_by_category = TERMS_BY_LANGUAGE[language]
    return terms_by_category.get(category, terms_by_category["default"])


def build_search_queries(
    osm_tags: dict,
    search_languages: list[str] | tuple[str, ...] = ("fr",),
) -> list[str]:
    name = get_osm_name(osm_tags)
    category = classify_polygon(osm_tags)
    queries = []

    for language in search_languages:
        terms = get_query_terms(category, language=language)
        queries.extend([f'"{name}" {term}' for term in terms])

    return list(dict.fromkeys(queries))

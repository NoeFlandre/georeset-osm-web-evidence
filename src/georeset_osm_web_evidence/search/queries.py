import re

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


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _title_context_value(value: str) -> str:
    lowercase_words = {"and", "of", "the"}
    words = []

    for word in value.split():
        if word.lower() in {"us", "usa"}:
            words.append("United States")
        elif word.lower() in lowercase_words:
            words.append(word.lower())
        else:
            words.append(word[:1].upper() + word[1:])

    return " ".join(words)


def clean_query_context_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = re.sub(r"[_/\-]+", " ", value).strip()
    cleaned = " ".join(cleaned.split())

    if not cleaned or cleaned.lower() in {"none", "nan", "<na>"}:
        return None

    return _title_context_value(cleaned)


def _expand_geographic_context_value(value: object) -> list[str]:
    if not isinstance(value, str):
        return []

    stripped = value.strip()
    if re.match(r"^us[/-]", stripped, flags=re.IGNORECASE):
        state = stripped.split("/", maxsplit=1)[-1].split("-", maxsplit=1)[-1]
        state_context = clean_query_context_value(state)
        return [context for context in [state_context, "United States"] if context]

    cleaned = clean_query_context_value(stripped)
    return [cleaned] if cleaned else []


def build_location_contexts(
    country: object = None,
    world_region: object = None,
    source_extract_id: object = None,
) -> list[str]:
    contexts = []

    for value in [country, source_extract_id]:
        contexts.extend(_expand_geographic_context_value(value))

    world_region_context = clean_query_context_value(world_region)
    if world_region_context:
        contexts.append(world_region_context)

    return _deduplicate(contexts)


def build_contextual_english_search_queries(
    osm_tags: dict,
    country: object = None,
    world_region: object = None,
    source_extract_id: object = None,
    polygon_category: str | None = None,
    max_queries: int = 4,
) -> list[str]:
    name = get_osm_name(osm_tags)
    category = polygon_category or classify_polygon(osm_tags)
    terms = get_query_terms(category, language="en")
    contexts = build_location_contexts(
        country=country,
        world_region=world_region,
        source_extract_id=source_extract_id,
    )
    primary_context = contexts[0] if contexts else None
    secondary_context = contexts[1] if len(contexts) > 1 else None
    category_label = category.replace("_", " ")
    query_candidates = []

    if primary_context:
        query_candidates.append(f'"{name}" "{primary_context}" {terms[0]}')

    if secondary_context:
        query_candidates.append(f'"{name}" "{secondary_context}" {terms[0]}')

    if primary_context:
        if len(terms) > 1:
            query_candidates.append(f'"{name}" "{primary_context}" {terms[1]}')
        query_candidates.append(f'"{name}" "{primary_context}" {category_label}')
        query_candidates.extend(
            f'"{name}" "{primary_context}" {term}' for term in terms[2:]
        )
    else:
        query_candidates.extend(f'"{name}" {term}' for term in terms)

    return _deduplicate(query_candidates)[:max_queries]


def build_location_topic_english_search_queries(
    osm_tags: dict,
    country: object = None,
    world_region: object = None,
    source_extract_id: object = None,
    polygon_category: str | None = None,
    max_queries: int = 4,
) -> list[str]:
    name = get_osm_name(osm_tags)
    category = polygon_category or classify_polygon(osm_tags)
    terms = get_query_terms(category, language="en")
    contexts = build_location_contexts(
        country=country,
        world_region=world_region,
        source_extract_id=source_extract_id,
    )
    location_context = contexts[0] if contexts else None

    if location_context is None:
        return _deduplicate(f'"{name}" "{term}"' for term in terms)[:max_queries]

    return _deduplicate(
        [f'"{name}" "{location_context}" "{term}"' for term in terms]
    )[:max_queries]


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

    return _deduplicate(queries)

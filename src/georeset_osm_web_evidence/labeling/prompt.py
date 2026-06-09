PROMPT_VERSION = "binary_remote_sensing_relevance_json_v2"
LOCATION_AWARE_PROMPT_VERSION = "binary_remote_sensing_location_relevance_json_v1"


def build_binary_label_prompt(sentence: str) -> str:
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Sentence must be a non-empty string")

    clean_sentence = " ".join(sentence.split())

    return f"""You are labeling sentences for a geospatial evidence dataset.

A positive sentence is a sentence that captures a characteristic of a geographic
location that is either directly visible in remote sensing imagery, or plausibly
correlated with characteristics visible in remote sensing imagery.

Label the sentence as:
- relevant: if it describes land cover, vegetation, water, terrain, agriculture,
  built structures, land use, conservation status, habitat, environmental
  condition, or another location characteristic useful for interpreting the
  place from geospatial/remote-sensing context.
- irrelevant: if it is generic website text, navigation, privacy/legal text,
  tourism logistics without place characteristics, business/service advertising,
  unrelated facts, or a sentence that does not describe the location itself.

Use only the sentence. Do not use the polygon name, URL, page title, or any
metadata to decide.

Sentence:
{clean_sentence}

Return only valid JSON with exactly this schema:
{{"label":"relevant"}} or {{"label":"irrelevant"}}"""


def build_location_aware_binary_label_prompt(
    sentence: str,
    polygon_name: str,
    location_context: str,
    polygon_category: str,
    page_title: str | None = None,
    search_query: str | None = None,
) -> str:
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Sentence must be a non-empty string")
    if not isinstance(polygon_name, str) or not polygon_name.strip():
        raise ValueError("Polygon name must be a non-empty string")

    clean_sentence = " ".join(sentence.split())
    clean_polygon_name = " ".join(polygon_name.split())
    clean_location_context = (
        " ".join(location_context.split())
        if isinstance(location_context, str) and location_context.strip()
        else "unknown"
    )
    clean_polygon_category = (
        " ".join(polygon_category.split())
        if isinstance(polygon_category, str) and polygon_category.strip()
        else "unknown"
    )
    clean_page_title = (
        " ".join(page_title.split())
        if isinstance(page_title, str) and page_title.strip()
        else "unknown"
    )
    clean_search_query = (
        " ".join(search_query.split())
        if isinstance(search_query, str) and search_query.strip()
        else "unknown"
    )

    return f"""You are labeling sentences for a geospatial evidence dataset.

Target polygon:
- Name: {clean_polygon_name}
- Location context: {clean_location_context}
- Polygon category: {clean_polygon_category}

A positive sentence must satisfy both conditions:
1. It describes the specific target polygon, or a place that is clearly the same
   geographic location as the target polygon.
2. It describes a characteristic useful for geospatial or remote-sensing
   interpretation, such as land cover, vegetation, water, wetlands, terrain,
   agriculture, crops, habitat, conservation status, environmental condition,
   built structures, or land use.

Label the sentence as irrelevant if it is only a generic fact about the topic,
describes another place, contains website/navigation/contact/legal text, or
mentions forests, wetlands, agriculture, biodiversity, or conservation without
describing the target polygon.

Source context:
- Page title: {clean_page_title}
- Search query: {clean_search_query}

Sentence:
{clean_sentence}

Return only valid JSON with exactly this schema:
{{"label":"relevant"}} or {{"label":"irrelevant"}}"""

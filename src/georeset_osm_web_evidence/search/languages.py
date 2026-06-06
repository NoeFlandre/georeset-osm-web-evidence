QUERY_LANGUAGE_BY_EXTRACT_ID = {
    "alsace": "fr",
    "extremadura": "es",
    "greenland": "da",
    "iraq": "ar",
    "libya": "ar",
    "sul": "pt",
}


def _row_value(row, column: str):
    if hasattr(row, "get"):
        return row.get(column)

    return getattr(row, column, None)


def resolve_query_local_language(row) -> str | None:
    source_extract_id = _row_value(row, "source_extract_id")
    configured_language = _row_value(row, "local_language")

    if isinstance(source_extract_id, str):
        override_language = QUERY_LANGUAGE_BY_EXTRACT_ID.get(source_extract_id)
        if override_language is not None:
            return override_language

    if isinstance(configured_language, str) and configured_language:
        return configured_language

    return None

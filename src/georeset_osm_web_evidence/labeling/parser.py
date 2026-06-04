VALID_LABELS = {"relevant", "irrelevant"}


def parse_binary_label_response(raw_response: str) -> str:
    if not isinstance(raw_response, str):
        raise ValueError("LLM response must be a string")

    label = raw_response.strip().lower()
    label = label.strip("`\"'").strip()
    label = label.rstrip(".").strip()
    label = label.strip("`\"'").strip()

    if label in VALID_LABELS:
        return label

    raise ValueError("Expected exactly one label: relevant or irrelevant")

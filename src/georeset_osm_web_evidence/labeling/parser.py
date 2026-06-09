import json


VALID_LABELS = {"relevant", "irrelevant"}


def parse_binary_label_response(raw_response: str) -> str:
    if not isinstance(raw_response, str):
        raise ValueError("LLM response must be a string")

    response = raw_response.strip()
    if response.startswith("```"):
        response_lines = response.splitlines()
        if response_lines and response_lines[0].startswith("```"):
            response_lines = response_lines[1:]
        if response_lines and response_lines[-1].strip() == "```":
            response_lines = response_lines[:-1]
        response = "\n".join(response_lines).strip()

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(
            'Expected JSON object with exactly one key: {"label": "relevant|irrelevant"}'
        ) from error

    if not isinstance(payload, dict) or set(payload) != {"label"}:
        raise ValueError(
            'Expected JSON object with exactly one key: {"label": "relevant|irrelevant"}'
        )

    label = payload["label"]
    if isinstance(label, str):
        label = label.strip().lower()

    if label in VALID_LABELS:
        return label

    raise ValueError("Expected label to be either relevant or irrelevant")

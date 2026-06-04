PROMPT_VERSION = "binary_remote_sensing_relevance_v1"


def build_binary_label_prompt(sentence: str) -> str:
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Sentence must be a non-empty string")

    clean_sentence = " ".join(sentence.split())

    return f"""You are labeling one sentence for a geospatial text classifier.

Label as relevant if the sentence captures a characteristic of a geographic location
that is visible via remote sensing, or correlated with characteristics visible via
remote sensing.

Label as irrelevant if the sentence does not describe such a location characteristic.

Sentence:
{clean_sentence}

Reply with exactly one word: relevant or irrelevant."""

from collections.abc import Callable

import pandas as pd

from georeset_osm_web_evidence.labeling.parser import parse_binary_label_response


def label_prompt_rows(
    prompt_df: pd.DataFrame,
    label_fn: Callable[[str], str],
) -> pd.DataFrame:
    result = prompt_df.copy()

    result["llm_label"] = None
    result["raw_response"] = None
    result["parse_error"] = None

    for index, row in result.iterrows():
        try:
            raw_response = label_fn(row["prompt"])
            result.at[index, "raw_response"] = raw_response
            result.at[index, "llm_label"] = parse_binary_label_response(raw_response)
        except Exception as error:
            result.at[index, "parse_error"] = (
                f"{error.__class__.__name__}: {error}"
            )

    return result

import pandas as pd


def append_unique_rows(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    return (
        pd.concat([existing_df, new_df], ignore_index=True)
        .drop_duplicates(subset=subset, keep="first")
        .reset_index(drop=True)
    )

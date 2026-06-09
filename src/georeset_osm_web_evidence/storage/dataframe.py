import logging
from pathlib import Path
from typing import Callable

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


def write_dataframe_artifact(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
    return dataframe


def load_or_build_dataframe(
    path: Path,
    stage_name: str,
    logger: logging.Logger,
    build: Callable[[], pd.DataFrame],
    reset: bool = False,
    load: Callable[[Path], pd.DataFrame] = pd.read_parquet,
    save: Callable[[pd.DataFrame, Path], None] | None = None,
) -> pd.DataFrame:
    if path.exists() and not reset:
        dataframe = load(path)
        logger.info("Loaded %s rows for %s from %s", len(dataframe), stage_name, path)
        return dataframe

    dataframe = build()
    if save is None:
        write_dataframe_artifact(dataframe, path)
    else:
        save(dataframe, path)
    logger.info("Saved %s rows for %s to %s", len(dataframe), stage_name, path)

    return dataframe

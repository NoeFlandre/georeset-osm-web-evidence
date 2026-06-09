from collections.abc import Callable
from pathlib import Path

import pandas as pd


def write_review_artifacts(
    review_df: pd.DataFrame,
    csv_output_path: str | Path,
    xlsx_output_path: str | Path,
    workbook_writer: Callable[[pd.DataFrame, str | Path], None],
) -> pd.DataFrame:
    csv_output_path = Path(csv_output_path)
    xlsx_output_path = Path(xlsx_output_path)

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(csv_output_path, index=False)
    workbook_writer(review_df, xlsx_output_path)

    return review_df

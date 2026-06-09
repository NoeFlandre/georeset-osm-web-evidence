from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.storage.dataframe import write_dataframe_artifact
from georeset_osm_web_evidence.web.quality import add_quality_metadata

DEFAULT_INPUT_PATH = Path("data/processed/evidence/page_text_sample.parquet")
DEFAULT_OUTPUT_PATH = Path(
    "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
)


def run_quality_metadata_build(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df_with_quality_metadata = add_quality_metadata(df)
    return write_dataframe_artifact(df_with_quality_metadata, output_path)


def main():
    run_quality_metadata_build(
        input_path=DEFAULT_INPUT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
    )
    print(f"Saved the dataframe with quality metadata at {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

from georeset_osm_web_evidence.web.quality import add_quality_metadata


def main():
    input_path = "data/processed/evidence/page_text_sample.parquet"
    output_path = Path(
        "data/processed/evidence/page_text_sample_with_quality_metadata.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df_with_quality_metadata = add_quality_metadata(df)

    df_with_quality_metadata.to_parquet(output_path, index=False)
    print(f"Saved the dataframe with quality metadata at {output_path}")


if __name__ == "__main__":
    main()

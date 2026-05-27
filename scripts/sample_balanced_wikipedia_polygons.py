import pandas as pd

from georeset_osm_web_evidence.storage.local import (
    load_geodataframe,
    save_geodataframe,
)


def main():
    input_path = "data/processed/osm_polygons_sample_batch300_wikipedia.parquet"
    output_path = (
        "data/processed/osm_polygons_sample100_wikipedia_balanced_50_50.parquet"
    )

    gdf = load_geodataframe(input_path)

    positives = gdf[gdf["has_wikipedia_articles"]].copy()
    negatives = gdf[~gdf["has_wikipedia_articles"]].copy()

    sample_size_per_class = 50

    if len(positives) < sample_size_per_class:
        raise ValueError(
            f"Needed {sample_size_per_class} polygons with a wikipedia articles but got {len(positives)}"
        )

    if len(negatives) < sample_size_per_class:
        raise ValueError(
            f"Needed {sample_size_per_class} polygons without a wikipedia article, got {len(negatives)}"
        )

    positives_sampled = positives.sample(
        n=sample_size_per_class,
        random_state=42,
    )

    negatives_sampled = negatives.sample(
        n=sample_size_per_class,
        random_state=42,
    )

    balanced_gdf = pd.concat(
        [positives_sampled, negatives_sampled],
        ignore_index=True,
    )

    balanced_gdf = balanced_gdf.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    save_geodataframe(balanced_gdf, output_path)
    print(f"Saved balanced geodataframe to {output_path}")


if __name__ == "__main__":
    main()

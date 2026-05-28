import time

import pandas as pd

from georeset_osm_web_evidence.storage.local import (
    load_geodataframe,
    save_geodataframe,
)
from georeset_osm_web_evidence.wikipedia.api import geosearch_wikipedia
from georeset_osm_web_evidence.wikipedia.spatial import (
    filter_articles_inside_polygon,
)


def find_articles_inside_polygons(
    row,
    language: str,
):
    candidates = geosearch_wikipedia(
        lat=row.centroid_lat,
        lon=row.centroid_lon,
        language=language,
        radius_m=10000,
        limit=10,
    )

    return filter_articles_inside_polygon(articles=candidates, polygon=row.geometry)


def main() -> None:
    batch_size = 300
    input_path = "data/raw/osm/named_polygon_candidates.parquet"
    output_path = "data/interim/wikipedia/named_polygon_candidates_wikipedia.parquet"

    gdf = load_geodataframe(input_path)
    batch_gdf = gdf.head(batch_size).copy()

    rows = []

    for index, row in enumerate(batch_gdf.itertuples(), start=1):
        print(f"Checking polygon {index} / {len(batch_gdf)}")
        articles_fr = find_articles_inside_polygons(row, language="fr")
        articles_en = find_articles_inside_polygons(row, language="en")

        rows.append(
            {
                "osm_type": row.osm_type,
                "osm_id": row.osm_id,
                "fr_wikipedia_articles": articles_fr,
                "en_wikipedia_articles": articles_en,
                "fr_wikipedia_articles_count": len(articles_fr),
                "en_wikipedia_articles_count": len(articles_en),
                "has_fr_wikipedia_articles": len(articles_fr) > 0,
                "has_en_wikipedia_articles": len(articles_en) > 0,
                "has_wikipedia_articles": bool(articles_fr or articles_en),
            }
        )

        time.sleep(0.5)

    wikipedia_df = pd.DataFrame(rows)

    enriched_gdf = batch_gdf.merge(
        wikipedia_df,
        on=["osm_type", "osm_id"],
        how="left",
    )

    save_geodataframe(enriched_gdf, output_path)

    print(f"Saved enriched geodataframe to {output_path}")
    print(f"Number of polygons having at least a wikipedia article:")
    print(enriched_gdf["has_wikipedia_articles"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

import pandas as pd

from georeset_osm_web_evidence.evidence.sentence_candidates import (
    build_sentence_candidate_dataframe,
    deduplicate_near_duplicate_sentence_candidates,
    filter_english_sentence_candidates,
    select_complete_sentence_candidates,
)
from georeset_osm_web_evidence.evidence.worldwide_pilot import attach_polygon_metadata


def build_english_sentence_candidates(
    page_text_with_quality_df: pd.DataFrame,
    pilot_gdf: pd.DataFrame,
    target_polygon_count: int,
    sentences_per_polygon: int,
    sentences_per_url: int,
) -> pd.DataFrame:
    sentence_df = build_sentence_candidate_dataframe(page_text_with_quality_df)
    sentence_df = attach_polygon_metadata(sentence_df, pilot_gdf)
    sentence_df = filter_english_sentence_candidates(sentence_df)
    sentence_df = deduplicate_near_duplicate_sentence_candidates(sentence_df)

    return select_complete_sentence_candidates(
        sentence_df,
        sentences_per_polygon=sentences_per_polygon,
        sentences_per_url=sentences_per_url,
        target_polygon_count=target_polygon_count,
    )

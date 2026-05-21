import geopandas as gpd


def sample_polygons(
    gdf: gpd.GeoDataFrame, sample_size: int = 100, random_state: int = 42
) -> gpd.GeoDataFrame:
    n = min(sample_size, len(gdf))
    return gdf.sample(
        n=n,
        random_state=random_state,
    ).copy()

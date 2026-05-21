import geopandas as gpd


def records_to_geodataframe(records: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )

import geopandas as gpd


def records_to_geodataframe(records: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )


def add_area_km2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    metric_gdf = gdf.to_crs("EPSG:2154")
    gdf["area_km2"] = metric_gdf.area / 1_000_000

    return gdf

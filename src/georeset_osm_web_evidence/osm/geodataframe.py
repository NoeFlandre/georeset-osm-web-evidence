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


def filter_by_area(
    gdf: gpd.GeoDataFrame,
    min_area_km2: float,
    max_area_km2: float,
) -> gpd.GeoDataFrame:
    return gdf[
        (gdf["area_km2"] >= min_area_km2) & (gdf["area_km2"] <= max_area_km2)
    ].copy()


def add_centroid_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    metric_gdf = gdf.to_crs("EPSG:2154")
    centroids = metric_gdf.centroid.to_crs("EPSG:4326")

    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y

    return gdf

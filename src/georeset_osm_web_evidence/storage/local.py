from pathlib import Path

import geopandas as gpd


def save_geodataframe(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_parquet(path, index=False)


def load_geodataframe(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)

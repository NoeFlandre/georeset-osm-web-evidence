import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from georeset_osm_web_evidence.storage.dataframe import (
    append_unique_rows,
    load_or_build_dataframe,
)


class DataFrameStorageTests(unittest.TestCase):
    def _silent_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    def test_append_unique_rows_preserves_existing_rows_before_new_duplicates(self):
        existing_df = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 1, "query": "forest", "url": "a"},
                {"osm_type": "way", "osm_id": 2, "query": "wetland", "url": "b"},
            ]
        )
        new_df = pd.DataFrame(
            [
                {"osm_type": "way", "osm_id": 2, "query": "wetland", "url": "updated"},
                {"osm_type": "way", "osm_id": 3, "query": "forest", "url": "c"},
            ]
        )

        result = append_unique_rows(
            existing_df,
            new_df,
            subset=["osm_type", "osm_id", "query"],
        )

        self.assertEqual(
            result.to_dict("records"),
            [
                {"osm_type": "way", "osm_id": 1, "query": "forest", "url": "a"},
                {"osm_type": "way", "osm_id": 2, "query": "wetland", "url": "b"},
                {"osm_type": "way", "osm_id": 3, "query": "forest", "url": "c"},
            ],
        )

    def test_load_or_build_dataframe_reuses_existing_artifact(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stage.parquet"
            existing_df = pd.DataFrame([{"value": "cached"}])
            existing_df.to_parquet(path, index=False)

            result = load_or_build_dataframe(
                path=path,
                stage_name="test stage",
                logger=self._silent_logger("test_load_existing_stage"),
                build=lambda: pd.DataFrame([{"value": "rebuilt"}]),
                reset=False,
            )

        self.assertEqual(result["value"].to_list(), ["cached"])

    def test_load_or_build_dataframe_rebuilds_when_reset_is_requested(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stage.parquet"
            pd.DataFrame([{"value": "cached"}]).to_parquet(path, index=False)

            result = load_or_build_dataframe(
                path=path,
                stage_name="test stage",
                logger=self._silent_logger("test_reset_stage"),
                build=lambda: pd.DataFrame([{"value": "rebuilt"}]),
                reset=True,
            )
            saved_df = pd.read_parquet(path)

        self.assertEqual(result["value"].to_list(), ["rebuilt"])
        self.assertEqual(saved_df["value"].to_list(), ["rebuilt"])


if __name__ == "__main__":
    unittest.main()

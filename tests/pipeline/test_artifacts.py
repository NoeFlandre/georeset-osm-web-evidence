import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from georeset_osm_web_evidence.pipeline.artifacts import (
    delete_artifacts,
    write_json_artifact,
)


class PipelineArtifactTests(unittest.TestCase):
    def test_write_json_artifact_creates_parent_and_logs_sorted_payload(self) -> None:
        logger = Mock(spec=logging.Logger)
        payload = {"z": 2, "a": {"b": 1}}

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "analysis.json"

            write_json_artifact(path, payload, logger=logger, log_label="Analysis")

            saved_payload = json.loads(path.read_text())
            saved_text = path.read_text()

        self.assertEqual(saved_payload, payload)
        self.assertTrue(saved_text.startswith('{\n  "a"'))
        logger.info.assert_called_once_with(
            "Analysis: %s",
            json.dumps(payload, sort_keys=True),
        )

    def test_delete_artifacts_removes_existing_files_and_ignores_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            existing_path = Path(temporary_directory) / "artifact.parquet"
            nested_path = Path(temporary_directory) / "nested" / "analysis.json"
            missing_path = Path(temporary_directory) / "missing.parquet"
            nested_path.parent.mkdir()
            existing_path.write_text("cached")
            nested_path.write_text("{}")

            delete_artifacts([existing_path, nested_path, missing_path])

            self.assertFalse(existing_path.exists())
            self.assertFalse(nested_path.exists())
            self.assertFalse(missing_path.exists())


if __name__ == "__main__":
    unittest.main()

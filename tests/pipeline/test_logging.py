import io
import logging
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from georeset_osm_web_evidence.pipeline.logging import configure_stage_logger


class PipelineLoggingTests(unittest.TestCase):
    def test_configure_stage_logger_writes_to_file_and_replaces_handlers(self) -> None:
        logger_name = "test_pipeline_stage_logger"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        old_handler = logging.NullHandler()
        logger.addHandler(old_handler)
        logger.setLevel(logging.WARNING)

        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "nested" / "run.log"
            stream = io.StringIO()

            with redirect_stderr(stream):
                configured_logger = configure_stage_logger(logger_name, log_path)
                configured_logger.info("pipeline logger works")
            for handler in configured_logger.handlers:
                handler.flush()

            log_text = log_path.read_text()

        self.assertIs(configured_logger, logger)
        self.assertEqual(configured_logger.level, logging.INFO)
        self.assertFalse(configured_logger.propagate)
        self.assertNotIn(old_handler, configured_logger.handlers)
        self.assertEqual(len(configured_logger.handlers), 2)
        self.assertIn("INFO pipeline logger works", log_text)
        self.assertIn("INFO pipeline logger works", stream.getvalue())


if __name__ == "__main__":
    unittest.main()

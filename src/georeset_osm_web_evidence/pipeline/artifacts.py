import json
import logging
from collections.abc import Iterable
from pathlib import Path


def delete_artifacts(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).unlink(missing_ok=True)


def write_json_artifact(
    path: str | Path,
    payload: dict,
    logger: logging.Logger | None = None,
    log_label: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if logger is not None and log_label is not None:
        logger.info(f"{log_label}: %s", json.dumps(payload, sort_keys=True))

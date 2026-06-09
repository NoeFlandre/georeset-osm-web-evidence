import os
from collections.abc import Mapping


def get_env_int(
    name: str,
    default: int,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if env is None else env
    return int(source.get(name, str(default)))


def get_env_float(
    name: str,
    default: float,
    *,
    env: Mapping[str, str] | None = None,
) -> float:
    source = os.environ if env is None else env
    return float(source.get(name, str(default)))


def get_env_flag(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if env is None else env
    return source.get(name, "0") == "1"

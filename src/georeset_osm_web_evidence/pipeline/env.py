import os
from collections.abc import Mapping


def get_env_int(
    name: str,
    default: int,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if env is None else env
    value = source.get(name, str(default))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def get_env_float(
    name: str,
    default: float,
    *,
    env: Mapping[str, str] | None = None,
) -> float:
    source = os.environ if env is None else env
    value = source.get(name, str(default))
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a float, got {value!r}") from error


def get_env_flag(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if env is None else env
    return source.get(name, "0") == "1"

"""Minimal pure-Python pytest compatibility shim for isolated test environments."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Optional, Type, Union

__version__ = "8.3.0"


class RaisesContext:
    """Context manager for pytest.raises."""

    def __init__(
        self,
        expected_exc: Union[Type[BaseException], tuple[Type[BaseException], ...]],
        match: Optional[str] = None,
    ) -> None:
        self.expected_exc = expected_exc
        self.match = match
        self.value: Optional[BaseException] = None

    def __enter__(self) -> "RaisesContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected_exc}")
        if not issubclass(exc_type, self.expected_exc):
            return False
        self.value = exc_val
        if self.match is not None:
            if not re.search(self.match, str(exc_val)):
                raise AssertionError(f"Pattern '{self.match}' not found in '{exc_val}'")
        return True


def raises(
    expected_exc: Union[Type[BaseException], tuple[Type[BaseException], ...]],
    *args: Any,
    **kwargs: Any,
) -> RaisesContext:
    """Assert that a block of code raises expected exception."""
    match = kwargs.get("match")
    return RaisesContext(expected_exc, match=match)


def fixture(*args: Any, **kwargs: Any) -> Callable:
    """Decorator to mark a fixture function."""
    def decorator(fn: Callable) -> Callable:
        fn.__pytest_fixture__ = True
        return fn

    if args and callable(args[0]):
        return decorator(args[0])
    return decorator


class _Mark:
    @staticmethod
    def parametrize(argnames: str, argvalues: list[Any], **kwargs: Any) -> Callable:
        def decorator(fn: Callable) -> Callable:
            fn.__pytest_parametrize__ = (argnames, argvalues)
            return fn
        return decorator


mark = _Mark()


def skip(reason: str = "") -> None:
    """Skip the current test."""
    raise _SkipTest(reason)


def fail(reason: str = "") -> None:
    """Explicitly fail the current test."""
    raise AssertionError(reason)


class _SkipTest(Exception):
    pass


class MonkeyPatch:
    """Simple monkeypatch implementation."""

    def __init__(self) -> None:
        self._undo: list[Callable[[], Any]] = []

    def setattr(self, target: Any, name: str, value: Any, raising: bool = True) -> None:
        if isinstance(target, str):
            mod_name, attr = target.rsplit(".", 1)
            target = sys.modules[mod_name]
            name = attr
        has_old = hasattr(target, name)
        old_val = getattr(target, name, None)
        setattr(target, name, value)
        self._undo.append(
            lambda: setattr(target, name, old_val) if has_old else delattr(target, name)
        )

    def setenv(self, name: str, value: str) -> None:
        old = os.environ.get(name)
        os.environ[name] = value
        self._undo.append(
            lambda: os.environ.pop(name, None) if old is None else os.environ.__setitem__(name, old)
        )

    def undo(self) -> None:
        for action in reversed(self._undo):
            try:
                action()
            except Exception:
                pass
        self._undo.clear()


def main(args: Optional[list[str]] = None) -> int:
    """Entry point compatible with pytest.main()."""
    from .__main__ import run_cli
    return run_cli(args)

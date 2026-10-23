"""CLI entry point for minimal pytest compatibility runner."""
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Callable

from . import MonkeyPatch, _SkipTest


def run_cli(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]

    rootdir = os.getcwd()
    test_paths: list[str] = []

    # Parse arguments
    for a in args:
        if not a.startswith("-"):
            test_paths.append(a)

    if not test_paths:
        if os.path.exists("tests"):
            test_paths.append("tests")
        else:
            test_paths.append(".")

    # Collect test files
    files_to_run: list[Path] = []
    for tp in test_paths:
        p = Path(tp)
        if p.is_file() and p.name.endswith(".py"):
            files_to_run.append(p.resolve())
        elif p.is_dir():
            for f in p.rglob("test_*.py"):
                files_to_run.append(f.resolve())

    files_to_run = sorted(list(set(files_to_run)))

    print("============================= test session starts ==============================")
    print(f"platform {sys.platform} -- Python {sys.version.split()[0]}")
    print(f"rootdir: {rootdir}")

    # Discover test items
    items: list[tuple[Path, str, Callable]] = []
    fixtures: dict[str, Callable] = {}

    for fpath in files_to_run:
        mod_name = fpath.stem
        spec = importlib.util.spec_from_file_location(mod_name, fpath)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod

        if str(fpath.parent) not in sys.path:
            sys.path.insert(0, str(fpath.parent))
        if rootdir not in sys.path:
            sys.path.insert(0, rootdir)

        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            print(f"ERROR collecting {fpath}: {exc}")
            traceback.print_exc()
            return 1

        # Register fixtures in module
        for fname, fval in inspect.getmembers(mod):
            if hasattr(fval, "__pytest_fixture__"):
                fixtures[fname] = fval

        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and name.startswith("Test"):
                try:
                    inst = obj()
                except Exception:
                    continue
                for mname, method in inspect.getmembers(inst, inspect.ismethod):
                    if mname.startswith("test_"):
                        items.append((fpath, f"{name}::{mname}", method))
            elif inspect.isfunction(obj) and name.startswith("test_"):
                items.append((fpath, name, obj))

    print(f"collected {len(items)} items\n")

    passed = 0
    failed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    start_time = time.time()

    current_file = None
    for fpath, name, func in items:
        rel = os.path.relpath(fpath, rootdir)
        if rel != current_file:
            if current_file is not None:
                sys.stdout.write("\n")
            current_file = rel
            sys.stdout.write(f"{rel} ")
            sys.stdout.flush()

        tmp_dir = tempfile.mkdtemp(prefix="pytest_tmp_")
        tmp_path = Path(tmp_dir)
        mp = MonkeyPatch()

        try:
            sig = inspect.signature(func)
            kwargs: dict[str, Any] = {}
            for param_name in sig.parameters:
                if param_name == "tmp_path":
                    kwargs["tmp_path"] = tmp_path
                elif param_name == "monkeypatch":
                    kwargs["monkeypatch"] = mp
                elif param_name in fixtures:
                    kwargs[param_name] = fixtures[param_name]()

            func(**kwargs)
            passed += 1
            sys.stdout.write(".")
            sys.stdout.flush()
        except _SkipTest:
            skipped += 1
            sys.stdout.write("s")
            sys.stdout.flush()
        except Exception:
            failed += 1
            sys.stdout.write("F")
            sys.stdout.flush()
            failures.append((f"{rel}::{name}", traceback.format_exc()))
        finally:
            mp.undo()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if current_file is not None:
        sys.stdout.write("\n")

    duration = time.time() - start_time

    if failures:
        print("\n=================================== FAILURES ===================================")
        for tname, tb in failures:
            print(f"___________________________________ {tname} ___________________________________")
            print(tb)

    summary_parts = []
    if passed:
        summary_parts.append(f"{passed} passed")
    if failed:
        summary_parts.append(f"{failed} failed")
    if skipped:
        summary_parts.append(f"{skipped} skipped")

    summary_str = ", ".join(summary_parts) if summary_parts else "0 passed"
    print(f"\n============================== {summary_str} in {duration:.2f}s ==============================")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(run_cli())

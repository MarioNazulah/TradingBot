"""Shared pytest configuration.

Proactive optional-dependency handling: the deep-RL stack (torch / SB3) is
heavy and platform-sensitive, and not every environment installs it (e.g. a
data-only checkout that just runs the Phase 2 indicator/QC tests). Rather than
letting a missing import abort collection of the *entire* suite, we scan every
test module and skip only those that import an unavailable heavy package.

This is self-maintaining: any new test module that imports torch/SB3 is handled
automatically, with no per-file `pytest.importorskip` needed. The explicit
guards already in individual modules remain as belt-and-suspenders.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

# Optional, heavy, environment-sensitive dependencies.
OPTIONAL_HEAVY = ("torch", "sb3_contrib", "stable_baselines3")

_missing = {name for name in OPTIONAL_HEAVY if importlib.util.find_spec(name) is None}

collect_ignore: list[str] = []

if _missing:
    _tests_dir = Path(__file__).parent
    for _path in sorted(_tests_dir.glob("test_*.py")):
        try:
            _tree = ast.parse(_path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        _top_level_imports: set[str] = set()
        for _node in ast.walk(_tree):
            if isinstance(_node, ast.Import):
                for _alias in _node.names:
                    _top_level_imports.add(_alias.name.split(".")[0])
            elif isinstance(_node, ast.ImportFrom) and _node.module:
                _top_level_imports.add(_node.module.split(".")[0])
        if _top_level_imports & _missing:
            collect_ignore.append(_path.name)

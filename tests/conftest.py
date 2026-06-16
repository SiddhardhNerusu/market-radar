"""Make the ``src/`` package importable in tests without an install step
(the project has no pyproject and is run via PYTHONPATH=src in production)."""
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

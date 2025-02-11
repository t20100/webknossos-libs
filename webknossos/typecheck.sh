#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck webknossos..."
poetry run python -m mypy -p webknossos

echo "Typecheck tests..."
poetry run python -m mypy -p tests

echo "Typecheck examples..."
poetry run python -m mypy -p examples

echo "Typecheck script-collection..."
poetry run python -m mypy -p script-collection

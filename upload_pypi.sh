#!/bin/bash
# Upload GuiXi to PyPI (Unix/Linux/macOS)

set -e

echo "=== GuiXi PyPI Upload Script ==="
echo ""

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "Python3 required but not installed."; exit 1; }
command -v twine >/dev/null 2>&1 || { echo "Twine not installed. Run: pip install twine"; exit 1; }
command -v build >/dev/null 2>&1 || { echo "Build not installed. Run: pip install build"; exit 1; }

# Get version from pyproject.toml
VERSION=$(grep -E '^version = ' pyproject.toml | sed 's/.*= *"\?\([^"]*\)"\?/\1/')
echo "Building version: $VERSION"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/
echo ""

# Build the package
echo "Building package..."
python3 -m build
echo ""

# Check the package
echo "Checking package..."
twine check dist/*
echo ""

# Upload to PyPI Test (optional, uncomment to use)
# echo "Uploading to PyPI Test..."
# twine upload --repository testpypi dist/*
# echo ""

# Upload to PyPI
echo "Uploading to PyPI..."
twine upload dist/*

echo ""
echo "=== Upload Complete ==="
echo "Package should be available at: https://pypi.org/project/guixi/$VERSION/"

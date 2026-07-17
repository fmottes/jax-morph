#!/usr/bin/env python3
"""Sync version from pyproject.toml to CITATION.cff."""

import re
from pathlib import Path

# Try tomllib (Python 3.11+) first, fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = PROJECT_ROOT / 'pyproject.toml'
CITATION_PATH = PROJECT_ROOT / 'CITATION.cff'


def get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml."""
    if tomllib is not None:
        with open(PYPROJECT_PATH, 'rb') as f:
            pyproject = tomllib.load(f)
        return pyproject['project']['version']
    else:
        # Fallback: use regex to extract version
        with open(PYPROJECT_PATH) as f:
            content = f.read()
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
        raise ValueError('Could not find version in pyproject.toml')


def update_citation_version(version: str) -> None:
    """Update version in CITATION.cff."""
    with open(CITATION_PATH) as f:
        content = f.read()

    # Replace the version line
    pattern = r'^version:\s*.*$'
    replacement = f'version: {version}'
    updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(CITATION_PATH, 'w') as f:
        f.write(updated_content)


def main() -> None:
    """Main function to sync version."""
    version = get_version_from_pyproject()
    print(f'Found version {version} in pyproject.toml')

    update_citation_version(version)
    print(f'Updated CITATION.cff to version {version}')


if __name__ == '__main__':
    main()

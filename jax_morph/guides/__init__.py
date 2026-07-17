"""Installed usage guides for jax-morph."""

from collections.abc import Iterator
from importlib import resources
from importlib.resources.abc import Traversable


def _iter_guide_names(root: Traversable, prefix: str = '') -> Iterator[str]:
    for entry in root.iterdir():
        if entry.is_dir():
            yield from _iter_guide_names(entry, f'{prefix}{entry.name}/')
        elif entry.is_file() and entry.name.endswith('.md'):
            yield f'{prefix}{entry.name.removesuffix(".md")}'


def list_guides() -> list[str]:
    """Return the names accepted by :func:`guide`."""
    return sorted(_iter_guide_names(resources.files(__package__)))


def guide(name: str = 'index') -> str:
    """Return an installed usage guide as Markdown text.

    Args:
        name: Guide name from :func:`list_guides`, without the ``.md`` suffix.

    Returns:
        The requested guide as Markdown text.

    Raises:
        ValueError: If ``name`` is not an available guide.
    """
    available = list_guides()
    if name not in available:
        raise ValueError(f'No guide {name!r}. Available guides: {available}')

    path = resources.files(__package__)
    parts = name.split('/')
    for part in (*parts[:-1], f'{parts[-1]}.md'):
        path = path.joinpath(part)
    return path.read_text(encoding='utf-8')


__all__ = ['guide', 'list_guides']

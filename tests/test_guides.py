from pathlib import Path

import pytest

import jax_morph as jxm

EXPECTED_GUIDES = [
    'basic-usage',
    'concepts',
    'core-abstractions',
    'extending',
    'index',
    'optimization/pathwise',
    'optimization/reinforce',
    'serialization',
]

DOC_LINKS = {
    'basic-usage.md': 'basic-usage.md',
    'concepts.md': 'concepts.md',
    'core-abstractions.md': 'core-abstractions.md',
    'extending.md': 'extending.md',
    'serialization.md': 'serialization.md',
    'optimization/pathwise.md': 'optimization/pathwise.md',
    'optimization/reinforce.md': 'optimization/reinforce.md',
}


def test_guides_are_public_and_discoverable():
    assert 'guides' in jxm.__all__
    assert jxm.guides.list_guides() == EXPECTED_GUIDES


def test_guide_reads_root_and_nested_markdown():
    assert jxm.guides.guide().startswith('# jax-morph usage guides')
    assert jxm.guides.guide('extending').startswith('# Writing a custom step')
    assert jxm.guides.guide('optimization/pathwise').startswith('# Pathwise training')


@pytest.mark.parametrize('name', ['missing', '../README', 'optimization/../../README'])
def test_guide_rejects_unknown_or_escaping_names(name):
    with pytest.raises(ValueError, match='Available guides'):
        jxm.guides.guide(name)


def test_docs_guide_pages_are_relative_symlinks_to_package_resources():
    root = Path(__file__).parents[1]
    for docs_path, guide_path in DOC_LINKS.items():
        link = root / 'docs' / docs_path
        target = root / 'jax_morph' / 'guides' / guide_path
        assert link.is_symlink()
        assert not link.readlink().is_absolute()
        assert link.resolve() == target.resolve()

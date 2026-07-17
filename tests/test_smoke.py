import jax_morph as jxm


def test_version():
    assert isinstance(jxm.__version__, str)

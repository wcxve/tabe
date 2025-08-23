from importlib.metadata import version

import tabe


def test_version():
    assert version('tabe') == tabe.__version__

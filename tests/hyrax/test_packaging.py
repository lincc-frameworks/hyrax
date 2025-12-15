import hyrax


def test_version():
    """Check to see that we can get the package version"""
    assert hyrax.__version__ is not None

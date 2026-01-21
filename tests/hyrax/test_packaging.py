import hyrax


def test_version():
    """Check to see that we can get the package version"""
    assert hyrax.__version__ is not None


def test_config_schemas_importable():
    """Ensure the config_schemas package is importable and functional."""
    from hyrax.config_schemas import BaseConfigModel

    class _Stub(BaseConfigModel):
        name: str

    model = _Stub(name="test")
    assert model.name == "test"

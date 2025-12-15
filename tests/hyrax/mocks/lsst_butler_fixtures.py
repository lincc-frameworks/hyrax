import pytest
from astropy.table import Table

SAMPLE_CATALOG_LENGTH = 50


def sample_catalog_data():
    """Create a sample astropy catalog for testing."""
    catalog_data = {
        "object_id": [1001 + i for i in range(SAMPLE_CATALOG_LENGTH)],
        "coord_ra": [150.0 + float(i) * 0.1 for i in range(SAMPLE_CATALOG_LENGTH)],
        "coord_dec": [2.0 + float(i) * 0.1 for i in range(SAMPLE_CATALOG_LENGTH)],
    }
    return catalog_data


@pytest.fixture(params=["fits", "hats"], scope="session")
def sample_catalog(request):
    """Create a sample astropy catalog for testing."""
    catalog_data = sample_catalog_data()
    catalog_type = request.param
    return catalog_type, convert_sample_catalog(catalog_type, catalog_data)


def convert_sample_catalog(catalog_type, catalog_data):
    """Convert the sample catalog to either hats or pandas"""
    if catalog_type == "fits":
        table = Table(catalog_data)

    elif catalog_type == "hats":
        import lsdb

        table = lsdb.from_dataframe(
            Table(catalog_data).to_pandas(), ra_column="coord_ra", dec_column="coord_dec"
        )

    return table


@pytest.fixture
def sample_catalog_saved(sample_catalog, tmp_path):
    """Give a path to a saved sample catalog"""
    catalog_type, table = sample_catalog
    catalog_path = save_catalog(catalog_type, table, tmp_path, "test_catalog")
    return catalog_type, catalog_path


def save_catalog(catalog_type, table, tmp_path, basename):
    """Save the catalog to a filesystem location"""
    catalog_path = tmp_path / f"{basename}.{catalog_type}"

    if catalog_type == "fits":
        table.write(catalog_path)
    elif catalog_type == "hats":
        table.to_hats(catalog_path)

    return catalog_path


@pytest.fixture
def lsst_config(sample_catalog_saved):
    """Create a basic configuration for LSSTDataset."""
    config_dict = {
        "data_set": {
            "butler_repo": "/fake/butler/repo",
            "butler_collection": "fake_collection",
            "skymap": "fake_skymap",
            "semi_height_deg": 0.01,
            "semi_width_deg": 0.01,
            "object_id_column_name": "object_id",
            "filters": ["g", "r", "i"],
            "transform": "tanh",
            "crop_to": [100, 100],
            "use_cache": False,
            "preload_cache": False,
        },
        "general": {
            "data_dir": "/tmp/test_data",
        },
    }

    catalog_type, catalog_path = sample_catalog_saved
    return set_catalog(config_dict, catalog_type, catalog_path)


def set_catalog(config_dict, catalog_type, catalog_path):
    """Set the catalog in a config dictionary depending on the type of catalog"""
    if catalog_type == "fits":
        config_dict["data_set"]["astropy_table"] = catalog_path
    elif catalog_type == "hats":
        config_dict["data_set"]["hats_catalog"] = catalog_path
    return config_dict

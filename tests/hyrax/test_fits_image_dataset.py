from pathlib import Path

import pytest
from torch import Size

import hyrax


@pytest.fixture(scope="function", params=[".astropy.csv", ".csv", ".fits", ".pq", ".votable"])
def test_hyrax_small_dataset_hscstars(request):
    """Fixture that gives a hyrax object configured to use the small_dataset_hscstars dataset

    Several common table formats are available for this dataset
    """
    this_file_dir = Path(__file__).absolute().parent
    catalog_file = this_file_dir / "test_data" / "small_dataset_hscstars" / f"star_cat_correct{request.param}"

    h = hyrax.Hyrax()
    h.config["model_inputs"] = {
        "data": {
            "dataset_class": "FitsImageDataSet",
            "data_location": str(catalog_file.parent),
            "primary_id_field": "object_id",
        },
    }
    h.config["data_set"]["filter_catalog"] = str(catalog_file)
    h.config["data_set"]["crop_to"] = [20, 20]

    object_id_column_name = "___object_id" if request.param == ".votable" else "# object_id"

    h.config["data_set"]["object_id_column_name"] = object_id_column_name
    h.config["data_set"]["filename_column_name"] = "star_filename"

    return h


def test_prepare(test_hyrax_small_dataset_hscstars):
    """Check that the hsc stars dataset was correctly read, and that basic access
    of FitsImageDataSet returns sane values
    """
    a = test_hyrax_small_dataset_hscstars.prepare()

    # Dataset has correct length
    assert len(a) == 10

    # All tensors are the correct size
    for d in a:
        assert d["data"]["image"].shape == Size([1, 20, 20])

    # Selected columns in the original catalog exist
    assert "ira" in a.metadata_fields("data")
    assert "idec" in a.metadata_fields("data")
    assert "SNR" in a.metadata_fields("data")

    # IDs are correct and in the correct order
    assert list(a.ids()) == [
        "36411452835238206",
        "36411452835248579",
        "36411452835249051",
        "36411452835250175",
        "36411457130203411",
        "36411457130204168",
        "36411457130206288",
        "36411457130214646",
        "36411457130215774",
        "36411457130216436",
    ]

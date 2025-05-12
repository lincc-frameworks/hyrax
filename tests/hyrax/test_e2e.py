import pytest

import hyrax


# Fixure to give us every model class which will be used for e2e esting
@pytest.fixture(scope="module", params=["HyraxAutoencoder"])
def model_class_name(request):
    """Fixture to generate the model names we want to test
    For this file all models must work with all data sources
    """
    return request.param


# Fixture to give us every dataset which will be used for testing
@pytest.fixture(
    scope="module",
    params=[
        ("HSCDataSet", "hsc1k"),
        ("HyraxCifarDataSet", None),
        ("HyraxCifarIterableDataSet", None),
        ("FitsImageDataSet", "hsc1k"),
    ],
)
def dataset_spec(request):
    """Fixture to generate all the Dataset class, sample data pairs
    Each of these must work with all models.
    """
    return request.param


# If the dataset requires a pre-download not performed by the class __init__
# it needs to be coded here.
@pytest.fixture(scope="module")
def tmp_dataset_path(tmp_path_factory, dataset_spec):
    """Fixture to download any needed sample data

    This is at module scope so it should only run once
    per run. for each data set, sample data pair.

    Additional sample data should use pooch and instance the directory name
    based on the sample_data name, because multiple data classes may parse
    the same sample dataset and we only want to download things once.
    """
    import pooch

    class_name, sample_data = dataset_spec

    if sample_data is None:
        return tmp_path_factory.mktemp(class_name)

    if sample_data == "hsc1k":
        tmp_path = tmp_path_factory.mktemp(sample_data)
        pooch.retrieve(
            # DOI for Example HSC dataset
            url="doi:10.5281/zenodo.14498536/hsc_demo_data.zip",
            known_hash="md5:1be05a6b49505054de441a7262a09671",
            fname="example_hsc_new.zip",
            path=tmp_path,
            processor=pooch.Unzip(extract_dir="."),
        )
        tmp_path = tmp_path / "hsc_8asec_1000"

    return tmp_path


# This gives a configured hyrax instance
@pytest.fixture(scope="function")
def hyrax_instance(tmp_dataset_path, dataset_spec, model_class_name, tmp_path):
    """Fixture to configure and initialize the hyrax instance"""
    h = hyrax.Hyrax()
    dataset_class_name, sample_data = dataset_spec
    h.config["general"]["data_dir"] = str(tmp_dataset_path)
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_set"]["name"] = dataset_class_name
    if dataset_class_name == "FitsImageDataSet" and sample_data == "hsc1k":
        h.config["data_set"]["filter_catalog"] = str(tmp_dataset_path / "manifest.fits")
        h.config["data_set"]["crop_to"] = [100, 100]
    h.config["model"]["name"] = model_class_name
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 128

    return h


@pytest.mark.slow
def test_init(hyrax_instance):
    """Test that the initialization fixtures function"""
    pass


@pytest.mark.slow
def test_getting_started(hyrax_instance):
    """Test that the basic flow we expect folks to run when
    getting started works
    """
    hyrax_instance.train()
    hyrax_instance.infer()
    hyrax_instance.umap()
    hyrax_instance.visualize()

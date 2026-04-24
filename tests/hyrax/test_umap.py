import os
import unittest.mock as mock

import numpy as np
import psutil
import pytest


class FakeUmap:
    """
    A Fake implementation of umap.UMAP which simply returns what is passed to it.
    This works with the loopback model and random dataset since they both output
    pairs of points, so the umap output is also pairs of points

    Install on a test like

    @mock.patch("umap.UMAP", FakeUmap)
    def test_blah():
        pass
    """

    def __init__(self, *args, **kwargs):
        print("Called FakeUmap init")
        # Store n_components from kwargs to match real UMAP behavior
        self.n_components = kwargs.get("n_components", 2)

    def fit(self, data):
        """We do nothing when fit on data. Prints are purely to help debug tests"""
        print("Called FakeUmap fit:")
        print(f"shape: {data.shape}")
        print(f"dtype: {data.dtype}")

    def transform(self, data):
        """We return our input when called to transform. Prints are purely to help debug tests"""
        print("Called FakeUmap transform:")
        print(f"shape: {data.shape}")
        print(f"dtype: {data.dtype}")
        return data


@mock.patch("umap.UMAP", FakeUmap)
def test_umap_order(loopback_inferred_hyrax):
    """Test that the order of data run through infer
    is correct in the presence of several splits
    """
    h, dataset, _ = loopback_inferred_hyrax

    dataset = dataset["infer"]

    umap_results = h.umap()
    umap_result_ids = umap_results.ids()
    original_dataset_ids = dataset.ids()

    data_shape = h.config["data_set"]["HyraxRandomDataset"]["shape"]

    for idx, result_id in enumerate(umap_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                break
        else:
            raise AssertionError("Failed to find a corresponding ID")

        umap_result = umap_results[idx].reshape(data_shape)

        print(f"orig idx: {dataset_idx}, umap idx: {idx}")
        print(f"orig data: {dataset[dataset_idx]}, umap data: {umap_result}")
        assert np.all(np.isclose(dataset[dataset_idx]["data"]["image"], umap_result))


@mock.patch("umap.UMAP", FakeUmap)
def test_umap_load(loopback_inferred_hyrax):
    """Test that umap loads a pre-existing model from a path and handles all error cases"""
    h, dataset, _ = loopback_inferred_hyrax
    dataset = dataset["infer"]

    # Calculate expected input dimensions
    infer_shape = np.array(h.config["data_set"]["HyraxRandomDataset"]["shape"])
    input_dim = int(np.prod(infer_shape))

    # Test successful loading
    fake_umap_instance = FakeUmap()
    fake_umap_instance._raw_data = np.zeros((100, input_dim))
    fake_umap_instance.n_components = 2

    data = psutil._common.bcat(f"/proc/{os.getpid()}/stat")
    print(data)

    with (
        mock.patch("builtins.open", mock.mock_open(read_data=b"fake_data")),
        mock.patch("pickle.load", return_value=fake_umap_instance),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        umap_result = h.umap(model_path="pretend_it_exists.pickle")
        assert umap_result is not None

    # Test missing UMAP model path raises FileNotFoundError
    with mock.patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(FileNotFoundError, match="UMAP model file not found"):
            h.umap(model_path="/nonexistent/path.pickle")

    # Test loading a non-UMAP object raises ValueError
    with (
        mock.patch("builtins.open", mock.mock_open(read_data=b"fake_data")),
        mock.patch("pickle.load", return_value=object()),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="loaded model is not a UMAP instance"):
            h.umap(model_path="/fake/path/umap.pickle")

    # Test loaded UMAP model with wrong input dimension raises ValueError
    fake_umap_wrong_input = FakeUmap()
    fake_umap_wrong_input._raw_data = np.zeros((100, input_dim + 1))
    fake_umap_wrong_input.n_components = 2

    with (
        mock.patch("builtins.open", mock.mock_open(read_data=b"fake_data")),
        mock.patch("pickle.load", return_value=fake_umap_wrong_input),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="input dimension of the loaded UMAP model"):
            h.umap(model_path="/fake/path/umap.pickle")

    # Test loaded UMAP model with wrong output dimension raises ValueError
    fake_umap_wrong_output = FakeUmap()
    fake_umap_wrong_output._raw_data = np.zeros((100, input_dim))
    fake_umap_wrong_output.n_components = 3

    with (
        mock.patch("builtins.open", mock.mock_open(read_data=b"fake_data")),
        mock.patch("pickle.load", return_value=fake_umap_wrong_output),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="output dimension of the loaded UMAP model"):
            h.umap(model_path="/fake/path/umap.pickle")

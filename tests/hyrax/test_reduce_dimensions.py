import unittest.mock as mock

import numpy as np
import pytest

from hyrax.verbs.reduction_algorithms.pca import PCA
from hyrax.verbs.reduction_algorithms.umap import UMAP


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
    """
    Test that the order of data run through infer
    is correct in the presence of several splits when using UMAP
    """
    h, dataset, _ = loopback_inferred_hyrax

    dataset = dataset["infer"]

    umap_results = h.reduce_dimensions(algorithm="umap")
    umap_result_ids = umap_results.ids()
    original_dataset_ids = dataset.ids()

    data_shape = h.config["data_set"]["HyraxRandomDataset"]["shape"]

    for idx, result_id in enumerate(umap_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                assert dataset_idx == idx
                break
        else:
            raise AssertionError("Failed to find a corresponding ID in original dataset")

        # Found the corresponding index in the original dataset, now check that the data matches
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

    with (
        mock.patch.object(UMAP, "_load_pickle", return_value=fake_umap_instance),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        umap_result = h.reduce_dimensions(algorithm="umap", model_path="pretend_model_exists.pickle")
        assert umap_result is not None

    # Test missing UMAP model path raises FileNotFoundError
    with mock.patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(FileNotFoundError, match="UMAP model file not found"):
            h.reduce_dimensions(algorithm="umap", model_path="not_a_file")

    # Test loading a non-UMAP object raises ValueError
    with (
        mock.patch.object(UMAP, "_load_pickle", return_value=object()),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="loaded model is not a UMAP instance"):
            h.reduce_dimensions(algorithm="umap", model_path="not_a_umap.pickle")

    # Test loaded UMAP model with wrong input dimension raises ValueError
    fake_umap_wrong_input = FakeUmap()
    fake_umap_wrong_input._raw_data = np.zeros((100, input_dim + 1))
    fake_umap_wrong_input.n_components = 2

    with (
        mock.patch.object(UMAP, "_load_pickle", return_value=fake_umap_wrong_input),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="input dimension of the loaded UMAP model"):
            h.reduce_dimensions(algorithm="umap", model_path="wrong_input_dim.pickle")

    # Test loaded UMAP model with wrong output dimension raises ValueError
    fake_umap_wrong_output = FakeUmap()
    fake_umap_wrong_output._raw_data = np.zeros((100, input_dim))
    fake_umap_wrong_output.n_components = 3

    with (
        mock.patch.object(UMAP, "_load_pickle", return_value=fake_umap_wrong_output),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="output dimension of the loaded UMAP model"):
            h.reduce_dimensions(algorithm="umap", model_path="wrong_output_dim.pickle")


class FakePCA:
    """
    A Fake implementation of sklearn.decomposition.PCA which simply returns what is passed to it.
    This works with the loopback model and random dataset since they both output
    pairs of points, so the pca output is also pairs of points

    Install on a test like

    @mock.patch("sklearn.decomposition.PCA", FakePCA)
    def test_blah():
        pass
    """

    def __init__(self, *args, **kwargs):
        print("Called FakePCA init")
        self.n_components = kwargs.get("n_components", 2)

    def fit(self, data):
        """We do nothing when fit on data. Prints are purely to help debug tests"""
        print("Called FakePCA fit:")
        print(f"shape: {data.shape}")
        print(f"dtype: {data.dtype}")

    def transform(self, data):
        """We return our input when called to transform. Prints are purely to help debug tests"""
        print("Called FakePCA transform:")
        print(f"shape: {data.shape}")
        print(f"dtype: {data.dtype}")
        return data


@mock.patch("sklearn.decomposition.PCA", FakePCA)
def test_pca_order(loopback_inferred_hyrax):
    """
    Test that the order of data run through infer
    is correct in the presence of several splits when using PCA
    """
    h, dataset, _ = loopback_inferred_hyrax

    dataset = dataset["infer"]

    pca_results = h.reduce_dimensions(algorithm="pca")
    pca_result_ids = pca_results.ids()
    original_dataset_ids = dataset.ids()

    data_shape = h.config["data_set"]["HyraxRandomDataset"]["shape"]

    for idx, result_id in enumerate(pca_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                assert dataset_idx == idx
                break
        else:
            raise AssertionError("Failed to find a corresponding ID in original dataset")

        # Found the corresponding index in the original dataset, now check that the data matches
        pca_result = pca_results[idx].reshape(data_shape)

        print(f"orig idx: {dataset_idx}, pca idx: {idx}")
        print(f"orig data: {dataset[dataset_idx]}, pca data: {pca_result}")
        assert np.all(np.isclose(dataset[dataset_idx]["data"]["image"], pca_result))


@mock.patch("sklearn.decomposition.PCA", FakePCA)
def test_pca_load(loopback_inferred_hyrax):
    """Test that pca loads a pre-existing model from a path and handles all error cases"""
    h, dataset, _ = loopback_inferred_hyrax
    dataset = dataset["infer"]

    # Calculate expected input dimensions
    infer_shape = np.array(h.config["data_set"]["HyraxRandomDataset"]["shape"])
    input_dim = int(np.prod(infer_shape))

    # Test successful loading
    fake_pca_instance = FakePCA()
    fake_pca_instance.n_features_in_ = input_dim
    fake_pca_instance.n_components_ = 2

    with (
        mock.patch.object(PCA, "_load_pickle", return_value=fake_pca_instance),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        pca_result = h.reduce_dimensions(algorithm="pca", model_path="pretend_model_exists.pickle")
        assert pca_result is not None

    # Test missing PCA model path raises FileNotFoundError
    with mock.patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(FileNotFoundError, match="PCA model file not found"):
            h.reduce_dimensions(algorithm="pca", model_path="not_a_file")

    # Test loading a non-PCA object raises ValueError
    with (
        mock.patch.object(PCA, "_load_pickle", return_value=object()),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="loaded model is not a PCA instance"):
            h.reduce_dimensions(algorithm="pca", model_path="not_a_pca.pickle")

    # Test loaded PCA model with wrong input dimension raises ValueError
    fake_pca_wrong_input = FakePCA()
    fake_pca_wrong_input.n_features_in_ = input_dim + 1
    fake_pca_wrong_input.n_components_ = 2

    with (
        mock.patch.object(PCA, "_load_pickle", return_value=fake_pca_wrong_input),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="input dimension of the loaded PCA model"):
            h.reduce_dimensions(algorithm="pca", model_path="wrong_input_dim.pickle")

    # Test loaded PCA model with wrong output dimension raises ValueError
    fake_pca_wrong_output = FakePCA()
    fake_pca_wrong_output.n_features_in_ = input_dim
    fake_pca_wrong_output.n_components_ = 3

    with (
        mock.patch.object(PCA, "_load_pickle", return_value=fake_pca_wrong_output),
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        with pytest.raises(ValueError, match="output dimension of the loaded PCA model"):
            h.reduce_dimensions(algorithm="pca", model_path="wrong_output_dim.pickle")


class FakeTSNE:
    """
    A Fake implementation of sklearn.manifold.TSNE which simply returns what is passed to it.
    This works with the loopback model and random dataset since they both output
    pairs of points, so the tsne output is also pairs of points

    Install on a test like

    @mock.patch("sklearn.manifold.TSNE", FakeTSNE)
    def test_blah():
        pass
    """

    def __init__(self, *args, **kwargs):
        print("Called FakeTSNE init")
        self.n_components = kwargs.get("n_components", 2)

    def fit_transform(self, data):
        """We return our input when called to fit_transform. Prints are purely to help debug tests"""
        print("Called FakeTSNE fit_transform:")
        print(f"shape: {data.shape}")
        print(f"dtype: {data.dtype}")
        return data


@mock.patch("sklearn.manifold.TSNE", FakeTSNE)
def test_tsne_order(loopback_inferred_hyrax):
    """
    Test that the order of data run through infer
    is correct in the presence of several splits when using t-SNE
    """
    h, dataset, _ = loopback_inferred_hyrax

    dataset = dataset["infer"]

    tsne_results = h.reduce_dimensions(algorithm="tsne")
    tsne_result_ids = tsne_results.ids()
    original_dataset_ids = dataset.ids()

    data_shape = h.config["data_set"]["HyraxRandomDataset"]["shape"]

    for idx, result_id in enumerate(tsne_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                assert dataset_idx == idx
                break
        else:
            raise AssertionError("Failed to find a corresponding ID in original dataset")

        # Found the corresponding index in the original dataset, now check that the data matches
        tsne_result = tsne_results[idx].reshape(data_shape)

        print(f"orig idx: {dataset_idx}, tsne idx: {idx}")
        print(f"orig data: {dataset[dataset_idx]}, tsne data: {tsne_result}")
        assert np.all(np.isclose(dataset[dataset_idx]["data"]["image"], tsne_result))

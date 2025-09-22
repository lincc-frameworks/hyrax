import unittest.mock as mock

import numpy as np


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

    umap_results = h.umap()
    umap_result_ids = list(umap_results.ids())
    original_dataset_ids = list(dataset.ids())

    if dataset.is_iterable():
        dataset = list(dataset)
        original_dataset_ids = np.array([str(s["object_id"]) for s in dataset])

    data_shape = h.config["data_set"]["HyraxRandomDataset"]["shape"]

    for idx, result_id in enumerate(umap_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                break
        else:
            raise AssertionError("Failed to find a corresponding ID")

        umap_result = umap_results[idx].cpu().numpy().reshape(data_shape)

        print(f"orig idx: {dataset_idx}, umap idx: {idx}")
        print(f"orig data: {dataset[dataset_idx]}, umap data: {umap_result}")
        assert np.all(np.isclose(dataset[dataset_idx]["data"]["image"], umap_result))

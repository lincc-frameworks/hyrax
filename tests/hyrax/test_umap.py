import pytest
import umap

# Example test data
umap_results = [...]  # replace with actual test data

def test_umap():
    data_shape = (10, 5)
    for idx in range(len(umap_results)):
        # Update line here
        umap_result = umap_results[idx].reshape(data_shape)
        assert umap_result.shape == data_shape

"""Tests for config["model"]["parallel_prepare_inputs"], which moves prepare_inputs into the dataloader."""

from unittest.mock import patch

import numpy as np
import torch

import hyrax
from hyrax.pytorch_ignite import (
    HyraxBatchContainer,
    create_save_batch_callback,
    dist_data_loader,
    setup_dataset,
)


class _RecordingModel:
    """Stub model whose prepare_inputs records every collated batch it receives."""

    def __init__(self):
        self.prepared = []

    def prepare_inputs(self, batch):
        self.prepared.append(batch)
        return batch["data"]["image"]


def _make_provider(tmp_path, parallel_prepare_inputs):
    """Return a config and a DataProvider over a 10-item HyraxRandomDataset."""
    config = hyrax.Hyrax().config
    config["model"]["parallel_prepare_inputs"] = parallel_prepare_inputs
    config["data_loader"]["batch_size"] = 4
    config["data_set"]["HyraxRandomDataset"]["size"] = 10
    config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    config["data_request"] = {
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path),
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }
    return config, setup_dataset(config, splits=("infer",))["infer"]


def test_dataloader_calls_prepare_inputs(tmp_path):
    """With parallel_prepare_inputs on, iterating the dataloader runs prepare_inputs on every
    batch and yields HyraxBatchContainers pairing its output with the batch's object ids."""
    config, provider = _make_provider(tmp_path, parallel_prepare_inputs=True)
    model = _RecordingModel()

    batches = list(dist_data_loader(provider, config, model=model))

    assert len(batches) == 3
    assert len(model.prepared) == 3
    for batch, collated in zip(batches, model.prepared):
        assert isinstance(batch, HyraxBatchContainer)
        np.testing.assert_array_equal(batch._hyrax_ids, collated["object_id"])
        np.testing.assert_array_equal(batch._hyrax_batch, collated["data"]["image"])
    all_ids = np.concatenate([batch._hyrax_ids for batch in batches])
    np.testing.assert_array_equal(all_ids, provider.ids())


def test_dataloader_skips_prepare_inputs_when_disabled(tmp_path):
    """With parallel_prepare_inputs off, the dataloader yields plain collated dicts."""
    config, provider = _make_provider(tmp_path, parallel_prepare_inputs=False)
    model = _RecordingModel()

    batches = list(dist_data_loader(provider, config, model=model))

    assert len(model.prepared) == 0
    assert all(isinstance(batch, dict) for batch in batches)


@patch("hyrax.datasets.result_factories.create_results_writer")
def test_save_batch_with_batch_container(mock_create_writer, tmp_path):
    """_save_batch writes the ids carried by a HyraxBatchContainer."""
    save_batch = create_save_batch_callback(tmp_path)
    ids = np.array(["a", "b"])
    batch = HyraxBatchContainer(_hyrax_ids=ids, _hyrax_batch=np.zeros((2, 3)))

    save_batch(batch, torch.ones(2, 3))

    written_ids, written_results = mock_create_writer.return_value.write_batch.call_args[0]
    np.testing.assert_array_equal(written_ids, ids)
    np.testing.assert_array_equal(written_results, np.ones((2, 3)))

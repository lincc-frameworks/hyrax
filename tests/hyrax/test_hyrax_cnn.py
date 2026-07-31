import numpy as np
import pytest
import torch

from hyrax.models.hyrax_cnn import HyraxCNN

BATCH_SIZE = 4
# HyraxCNN runs images through two Conv2d(kernel=5) + MaxPool2d(kernel=2, stride=2)
# stages, which requires at least an 18x18 input to avoid collapsing to a 0x0
# tensor before the final layers. 20x20 is the smallest round size above that
# floor, so we use it here instead of a truly minimal test image.
IMAGE_SHAPE = (1, 20, 20)  # (channels, width, height)
OUTPUT_CLASSES = 2


def make_model_config(output_classes=OUTPUT_CLASSES):
    """Minimal config dict containing the sections HyraxCNN needs from
    `hyrax_model` to build its optimizer/criterion/scheduler directly,
    without going through a full `hyrax.Hyrax()` instance."""
    return {
        "model": {"HyraxCNN": {"output_classes": output_classes}},
        "criterion": {"name": "torch.nn.CrossEntropyLoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01, "momentum": 0.9},
        "scheduler": {"name": None},
    }


def make_data_sample():
    """A stand-in for the `data_sample` HyraxCNN uses at init time to size its
    layers dynamically. Only the image tensor's shape is inspected."""
    return (torch.zeros(BATCH_SIZE, *IMAGE_SHAPE), torch.zeros(BATCH_SIZE, dtype=torch.long))


def make_hyrax_batch(batch_size=BATCH_SIZE, seed=0):
    """Build a batch in Hyrax's collated "dict of lists" format: a single
    dict of keys where each value is a numpy array holding every datapoint
    in the batch concatenated together. This is the shape of data that
    `prepare_inputs` receives from the real data pipeline.
    """
    rng = np.random.default_rng(seed)
    images = rng.random((batch_size, *IMAGE_SHAPE)).astype(np.float32)
    labels = rng.integers(0, 2, size=batch_size)

    return {
        "data": {
            "index": np.arange(batch_size),
            "object_id": np.array([str(i) for i in range(batch_size)]),
            "image": images,
            "label": labels,
        }
    }


def to_tensor_batch(prepared_inputs):
    """Convert the (image, label) numpy arrays returned by `prepare_inputs`
    into the torch tensors that `train_batch`/`validate_batch`/etc. expect.
    In the real pipeline this conversion is done by Hyrax automatically.
    """
    image, label = prepared_inputs
    return torch.from_numpy(image), torch.from_numpy(label)


@pytest.fixture
def model():
    """A freshly constructed HyraxCNN sized for 1x5x5 images and a binary
    (0/1) label, matching the dataset format described for this model."""
    return HyraxCNN(config=make_model_config(), data_sample=make_data_sample())


def test_prepare_inputs_extracts_image_and_label():
    """`prepare_inputs` should pull `image` and `label` out of the
    dict-of-lists batch format and return them as float32/int64 arrays."""
    hyrax_batch = make_hyrax_batch()

    image, label = HyraxCNN.prepare_inputs(hyrax_batch)

    assert image.shape == (BATCH_SIZE, *IMAGE_SHAPE)
    assert image.dtype == np.float32
    assert label.shape == (BATCH_SIZE,)
    assert label.dtype == np.int64
    np.testing.assert_array_equal(image, hyrax_batch["data"]["image"])
    np.testing.assert_array_equal(label, hyrax_batch["data"]["label"])


def test_prepare_inputs_raises_without_data_key():
    """`prepare_inputs` should fail loudly if the pipeline hands it a batch
    that isn't in the expected `{"data": {...}}` shape."""
    with pytest.raises(RuntimeError, match="data"):
        HyraxCNN.prepare_inputs({"not_data": {}})


def test_forward_produces_expected_output_shape(model):
    """forward() should return one class-score row per batch item."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxCNN.prepare_inputs(hyrax_batch))

    output = model(batch)

    assert output.shape == (BATCH_SIZE, OUTPUT_CLASSES)


def test_train_batch_runs_and_updates_weights(model):
    """train_batch() should return a loss and leave the model's weights changed."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxCNN.prepare_inputs(hyrax_batch))

    weights_before = model.fc3.weight.clone()
    result = model.train_batch(batch)

    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert not torch.equal(weights_before, model.fc3.weight)


def test_validate_batch_runs_without_updating_weights(model):
    """validate_batch() should return a loss without changing the model's weights."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxCNN.prepare_inputs(hyrax_batch))

    weights_before = model.fc3.weight.clone()
    result = model.validate_batch(batch)

    assert "loss" in result
    torch.testing.assert_close(weights_before, model.fc3.weight)


def test_test_batch_runs_without_updating_weights(model):
    """test_batch() should return a loss without changing the model's weights."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxCNN.prepare_inputs(hyrax_batch))

    weights_before = model.fc3.weight.clone()
    result = model.test_batch(batch)

    assert "loss" in result
    torch.testing.assert_close(weights_before, model.fc3.weight)


def test_infer_batch_returns_model_output(model):
    """infer_batch() should return one class-score row per batch item."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxCNN.prepare_inputs(hyrax_batch))

    output = model.infer_batch(batch)

    assert output.shape == (BATCH_SIZE, OUTPUT_CLASSES)


def test_train_batch_is_reproducible_with_fixed_seed():
    """Stretch goal: given the same rng seed for model initialization and the
    same input batch, training a single batch should be fully reproducible.
    """
    hyrax_batch = make_hyrax_batch(seed=42)
    batch = to_tensor_batch(HyraxCNN.prepare_inputs(hyrax_batch))

    torch.manual_seed(1234)
    model_a = HyraxCNN(config=make_model_config(), data_sample=make_data_sample())
    loss_a = model_a.train_batch(batch)["loss"]

    torch.manual_seed(1234)
    model_b = HyraxCNN(config=make_model_config(), data_sample=make_data_sample())
    loss_b = model_b.train_batch(batch)["loss"]

    np.testing.assert_almost_equal(loss_a, loss_b)


@pytest.fixture(scope="function")
def hyrax_cnn_pipeline(tmp_path_factory):
    """A full `hyrax.Hyrax()` instance wired up to train/validate/test/infer
    HyraxCNN end-to-end against a `HyraxRandomDataset` configured to produce
    1x5x5 single-channel images with a 0/1 label, i.e. exactly the data
    format `prepare_inputs`/`train_batch`/etc. are meant to handle.
    """
    import hyrax

    results_dir = tmp_path_factory.mktemp("hyrax_cnn_pipeline")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxCNN"
    h.config["model"]["HyraxCNN"]["output_classes"] = OUTPUT_CLASSES
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = BATCH_SIZE
    h.config["general"]["results_dir"] = str(results_dir)
    h.config["general"]["dev_mode"] = True

    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "test": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data_infer")),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["split"] = {"train": 0.6, "validate": 0.2, "test": 0.2}
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = list(IMAGE_SHAPE)
    h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = [0, 1]

    return h


def test_hyrax_cnn_train_validate_through_pipeline(hyrax_cnn_pipeline):
    """Integration test: training should run `train_batch` and
    `validate_batch` for every batch produced by the real Hyrax data
    pipeline without error."""
    h = hyrax_cnn_pipeline
    model = h.train()

    assert model is not None


def test_hyrax_cnn_test_through_pipeline(hyrax_cnn_pipeline):
    """Integration test: the `test` verb should run `test_batch` for every
    batch produced by the real Hyrax data pipeline without error, and
    return a ResultDataset."""
    from hyrax.datasets.result_dataset import ResultDataset

    h = hyrax_cnn_pipeline
    h.train()  # produce weights for the test verb to load
    result = h.test()

    assert isinstance(result, ResultDataset)


def test_hyrax_cnn_infer_through_pipeline(hyrax_cnn_pipeline):
    """Integration test: the `infer` verb should run `infer_batch` for every
    batch produced by the real Hyrax data pipeline without error."""
    h = hyrax_cnn_pipeline
    h.train()  # produce weights for the infer verb to load
    inference_results = h.infer()

    assert inference_results is not None
    assert len(inference_results) > 0

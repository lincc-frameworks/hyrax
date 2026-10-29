"""Tests for how the built-in image models size themselves from ``[model.<Name>].input_shape``."""

import copy

import pytest
import torch

import hyrax
from hyrax.models.hyrax_autoencoder import HyraxAutoencoder
from hyrax.models.hyrax_autoencoderv2 import HyraxAutoencoderV2
from hyrax.models.hyrax_cnn import HyraxCNN
from hyrax.models.image_dcae import ImageDCAE
from hyrax.models.model_utils import read_input_shape

IMAGE_MODELS = [
    pytest.param("HyraxCNN", HyraxCNN, id="HyraxCNN"),
    pytest.param("HyraxAutoencoder", HyraxAutoencoder, id="HyraxAutoencoder"),
    pytest.param("HyraxAutoencoderV2", HyraxAutoencoderV2, id="HyraxAutoencoderV2"),
    pytest.param("ImageDCAE", ImageDCAE, id="ImageDCAE"),
]

BAD_SHAPES = [
    pytest.param(False, id="false-sentinel"),
    pytest.param([3, 32], id="too-short"),
    pytest.param([1, 3, 32, 32], id="too-long"),
    pytest.param([3, "32", 32], id="string"),
    pytest.param([3, 0, 32], id="zero"),
    pytest.param([3, -32, 32], id="negative"),
    pytest.param([3, True, 32], id="bool"),
    pytest.param(32, id="scalar"),
]


@pytest.fixture(scope="module")
def default_config():
    """The default Hyrax configuration, loaded once for the module."""
    return hyrax.Hyrax().config


@pytest.fixture
def config(default_config):
    """A fresh copy of the default configuration for each test to mutate."""
    return copy.deepcopy(default_config)


def _batch_for(model_name, shape, batch_size=2):
    """A batch of zeros in the form the model's forward expects."""
    images = torch.zeros(batch_size, *shape)
    if model_name == "HyraxCNN":
        return images, torch.zeros(batch_size, dtype=torch.long)
    return images


@pytest.mark.parametrize("model_name,model_cls", IMAGE_MODELS)
def test_default_input_shape_builds_the_model(config, model_name, model_cls):
    """Every image model constructs from the shipped default config and accepts matching images."""
    model = model_cls(config)

    assert model.input_shape == (3, 32, 32)
    assert model(_batch_for(model_name, (3, 32, 32))).shape[0] == 2


@pytest.mark.parametrize("model_name,model_cls", IMAGE_MODELS)
@pytest.mark.parametrize("bad_shape", BAD_SHAPES)
def test_bad_input_shape_raises_naming_the_key(config, model_name, model_cls, bad_shape):
    """A malformed input_shape fails at construction with the config key in the message."""
    config["model"][model_name]["input_shape"] = bad_shape

    with pytest.raises(ValueError, match=rf"config\['model'\]\['{model_name}'\]\['input_shape'\]"):
        model_cls(config)


def test_read_input_shape_accepts_a_tuple(config):
    """A tuple, as tests and notebooks tend to write, is as good as a TOML list."""
    config["model"]["HyraxCNN"]["input_shape"] = (1, 20, 20)

    assert read_input_shape(config, "HyraxCNN") == (1, 20, 20)

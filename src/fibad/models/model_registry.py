MODEL_REGISTRY = {}


def fibad_model(cls):
    """Decorator to register a model with the model registry.

    Returns
    -------
    type
        The original, unmodified class.
    """
    update_model_registry(cls.__name__, cls)
    return cls


def update_model_registry(name: str, model_class: type):
    """Add a model to the model registry.

    Parameters
    ----------
    name : str
        The name of the model.
    model_class : type
        The model class.
    """

    MODEL_REGISTRY.update({name: model_class})

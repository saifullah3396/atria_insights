import inspect
import re
from functools import wraps
from importlib import import_module


def filter_kwargs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return _filter_func_kwargs(func, *args, **kwargs)

    return wrapper


def _filter_func_kwargs(func, *args, **kwargs):
    # Get the signature of the function
    sig = inspect.signature(func)
    # Extract the parameter names from the function signature
    valid_params = set(sig.parameters.keys())
    # Filter kwargs to only include valid parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    # Call the function with the filtered kwargs
    return func(*args, **filtered_kwargs)


def _get_parent_module(module_name: str):
    # Check if the module name has a parent (e.g., 'parent.child' -> 'parent')
    parent_module_name = (
        module_name.rsplit(".", 1)[0] if "." in module_name else module_name
    )

    return parent_module_name


def _resolve_module_from_path(module_path: str):
    path = module_path.rsplit(".", 1)
    if len(path) == 1:
        raise ValueError(
            f"Invalid module path: {module_path}. It should be in the form 'module_name.class_name'."
        )
    module_name, class_name = path
    module = import_module(module_name)
    return getattr(module, class_name)


def _convert_to_snake_case(s: str) -> str:
    """
    Converts a camel case string to snake case.

    Args:
        s (str): The camel case string.

    Returns:
        str: The underscored lower case string.
    """
    return re.sub(r"([A-Z])", r"_\1", s).lower().lstrip("_")


def _flatten_dict(d, parent_key="", sep="_"):
    flattened = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value
    return flattened

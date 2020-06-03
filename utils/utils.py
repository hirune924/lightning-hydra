import importlib
from typing import Any

import numpy as np
import collections
import torch


# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def preds_rounder(test_preds, num_class):
    # print(np.floor(np.clip(test_preds + 0.5, 0, num_class)))
    test_preds = np.floor(np.clip(test_preds + 0.5, 0, num_class - 1))
    return test_preds


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_pytorch_model(ckpt_name, model):
    state_dict = torch.load(ckpt_name)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("model."):
            name = name.replace("model.", "")  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

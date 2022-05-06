import json
import random
from typing import TypeVar, Callable, Dict

import numpy as np
import torch
import transformers

T = TypeVar("T")


def load_as_object(json_path: str, instantiator: Callable[[Dict], T]) -> T:
    with open(json_path) as f:
        config_text = f.read()

    # load json config file as object
    config_dict = json.loads(config_text)
    return instantiator(config_dict)


def select_first_available(values):
    return next(item for item in values if item is not None)


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)

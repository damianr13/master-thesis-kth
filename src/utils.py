import json
from typing import TypeVar, Callable, Dict

T = TypeVar("T")


def load_as_object(json_path: str, instantiator: Callable[[Dict], T]) -> T:
    with open(json_path) as f:
        config_text = f.read()

    # load json config file as object
    config_dict = json.loads(config_text)
    return instantiator(config_dict)
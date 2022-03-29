import os
from abc import ABC
from abc import abstractmethod
from typing import Callable, Generic, TypeVar, Dict

import pandas as pd
from pandas import DataFrame

from radamian import utils
from radamian.preprocess.configs import BasePreprocConfig

T = TypeVar("T", bound=BasePreprocConfig)


class BasePreprocessor(Generic[T], ABC):
    def __init__(self, config_path: str, config_instantiator: Callable[[Dict], T]):
        self.config = utils.load_as_object(config_path, config_instantiator)

    @abstractmethod
    def _preprocess_one(self, df: DataFrame) -> DataFrame:
        pass

    def preprocess(self):
        original_location = self.config.original_location
        target_location = self.config.target_location
        if not os.path.exists(target_location):
            os.makedirs(target_location)

        for source, target in self.config.split_files.items():
            part_refs = pd.read_csv(os.path.join(original_location, source))
            part_stand = self._preprocess_one(part_refs)

            part_stand.to_csv(os.path.join(target_location, target))

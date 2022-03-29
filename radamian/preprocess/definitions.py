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

    def _preprocess_one(self, df: DataFrame) -> DataFrame:
        return df

    def _preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        return df_for_location

    def preprocess(self):
        original_location = self.config.original_location
        target_location = self.config.target_location
        if not os.path.exists(target_location):
            os.makedirs(target_location)

        df_for_location: Dict[str, DataFrame] = {}
        for source, target in self.config.split_files.items():
            part_refs = self.read_one_split_file(os.path.join(original_location, source))
            part_stand = self._preprocess_one(part_refs)

            df_for_location[target] = part_stand

        df_for_location = self._preprocess_all(df_for_location)
        for location, df in df_for_location.items():
            df.to_csv(os.path.join(target_location, location))

    @staticmethod
    def read_one_split_file(path):
        return pd.read_csv(path)

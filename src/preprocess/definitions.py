import os
import re
from abc import ABC
from typing import Callable, Generic, TypeVar, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame

from src import utils
from src.preprocess.configs import BasePreprocConfig

T = TypeVar("T", bound=BasePreprocConfig)


class BasePreprocessor(Generic[T], ABC):
    def __init__(self, config_path: str, config_instantiator: Callable[[Dict], T]):
        self.config = utils.load_as_object(config_path, config_instantiator)

    def extract_relevant_columns_one(self, df: DataFrame) -> DataFrame:
        relevant_column_names = [f'left_{c}' for c in self.config.relevant_columns] + \
                                [f'right_{c}' for c in self.config.relevant_columns] + \
                                ['label']
        return df[relevant_column_names]

    def extract_relevant_columns(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        return {location: self.extract_relevant_columns_one(df) for location, df in df_for_location.items()}

    def preprocess_one(self, df: DataFrame) -> DataFrame:
        return df

    def preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        return df_for_location

    def preprocess(self, original_location: str = None, target_location: str = None):
        original_location = utils.select_first_available([self.config.original_location, original_location])
        target_location = utils.select_first_available([self.config.target_location, target_location])

        target_files_exist = np.all([os.path.exists(os.path.join(target_location, target))
                                     for target in self.config.split_files.values()])
        if target_files_exist:
            print(f'All files are already in place, skipping step for target location: {target_location}')
            return

        if not os.path.exists(target_location):
            os.makedirs(target_location)

        df_for_location: Dict[str, DataFrame] = {}
        for source, target in self.config.split_files.items():
            part_refs = self.read_one_split_file(os.path.join(original_location, source))
            part_stand = self.preprocess_one(part_refs)

            df_for_location[target] = part_stand

        df_for_location = self.preprocess_all(df_for_location)
        df_for_location = self.extract_relevant_columns(df_for_location)

        for location, df in df_for_location.items():
            df.to_csv(os.path.join(target_location, location), index=False)

    @staticmethod
    def read_one_split_file(path) -> DataFrame:
        return pd.read_csv(path)


class NoColumnSelectionBasePreprocessor(Generic[T], BasePreprocessor[T]):
    def __init__(self, config_path: str, config_instantiator: Callable[[Dict], T]) -> None:
        super(NoColumnSelectionBasePreprocessor, self).__init__(config_path, config_instantiator)

    def extract_relevant_columns_one(self, df: DataFrame) -> DataFrame:
        return df  # no column selection


class TransformerLMBasePreprocessor(Generic[T], BasePreprocessor[T]):
    def __init__(self, config_path: str, config_instantiator: Callable[[Dict], T]):
        super(TransformerLMBasePreprocessor, self).__init__(config_path=config_path,
                                                            config_instantiator=config_instantiator)

    def extract_relevant_columns_one(self, df: DataFrame) -> DataFrame:
        return df[['left_text', 'right_text', 'label']]

    @staticmethod
    def _apply_preprocessing(text: str, max_length: int):
        if not isinstance(text, str):
            return text
        text = re.sub(re.compile("\\s+"), " ", text)
        text = text.strip()

        return ' '.join(text.split(' ')[:max_length])

    def _get_relevant_columns(self) -> List[str]:
        return self.config.relevant_columns

    def preprocess_one(self, df: DataFrame) -> DataFrame:
        textual_columns = self._get_relevant_columns()

        left_textual_column = {f'left_{c}': c for c in textual_columns}
        right_textual_column = {f'right_{c}': c for c in textual_columns}

        def concat_with_annotation(sample, columns: Dict[str, str]):
            return ' '.join([f'[COL] {v} [VAL] {self._apply_preprocessing(sample[k], self.config.column_lengths[v])}'
                             for k, v in columns.items()])

        df['left_text'] = df.agg(lambda x: concat_with_annotation(x, left_textual_column), axis=1)
        df['right_text'] = df.agg(lambda x: concat_with_annotation(x, right_textual_column), axis=1)

        return df

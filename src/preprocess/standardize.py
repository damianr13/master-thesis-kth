import os
from typing import Dict

import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import RelationalStandardizerConfig, WDCStandardizerConfig
from src.preprocess.definitions import BasePreprocessor


class RelationalDatasetStandardizer(BasePreprocessor[RelationalStandardizerConfig]):
    """
    Applies standardization operation for datasets with a "relational format".

    These datasets have the 2 data sources in separate files, and 3 other files (train, test, valid) defining the
    train / test / valid split by referencing those 2 data sources. Instead, we need the 3 files to contain all the
    required data for working with the model
    """

    def __init__(self, config_path):
        super(RelationalDatasetStandardizer, self).__init__(config_path=config_path,
                                                            config_instantiator=RelationalStandardizerConfig.parse_obj)

        self.source_a = pd.read_csv(os.path.join(self.config.original_location, 'tableA.csv'))
        self.source_b = pd.read_csv(os.path.join(self.config.original_location, 'tableB.csv'))

    def _preprocess_one(self, refs_datas: DataFrame) -> DataFrame:
        result = refs_datas

        # merge the reference table with the data sources
        result = result.merge(self.source_a.add_prefix('left_'), left_on='ltable_id', right_on='left_id', how='inner')
        result = result.merge(self.source_b.add_prefix('right_'), left_on='rtable_id', right_on='right_id', how='inner')

        relevant_column_names = [f'left_{c}' for c in self.config.relevant_columns] + \
                                [f'right_{c}' for c in self.config.relevant_columns] + \
                                ['label']
        return result[relevant_column_names]


class WDCDatasetStandardizer(BasePreprocessor[WDCStandardizerConfig]):
    def __init__(self, config_path):
        super(WDCDatasetStandardizer, self).__init__(config_path=config_path,
                                                     config_instantiator=WDCStandardizerConfig.parse_obj)

    @staticmethod
    def read_one_split_file(path):
        return pd.read_json(path, lines=True)

    def __apply_correct_names(self, df: DataFrame):
        column_name_map = {}
        for name in self.config.relevant_columns:
            column_name_map[name + "_left"] = "left_" + name
            column_name_map[name + "_right"] = "right_" + name

        df = df.rename(columns=column_name_map)
        return df[list(column_name_map.values()) + ['label']]

    def _preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        train_valid_df = df_for_location[self.config.intermediate_train_valid_name]
        train_valid_split = pd.read_csv(os.path.join(self.config.original_location, self.config.train_valid_split_file))

        train_valid_split[['id_left', 'id_right']] = train_valid_split['pair_id']\
            .str.split('#', 1, expand=True).astype(int)
        valid_df = train_valid_df[train_valid_df.set_index(['id_left', 'id_right']).index.isin(
            train_valid_split.set_index(['id_left', 'id_right']).index)]

        train_df = train_valid_df[~train_valid_df.set_index(['id_left', 'id_right']).index.isin(
            valid_df.set_index(['id_left', 'id_right']).index)]
        result = {k: v for k, v in df_for_location.items() if k != self.config.intermediate_train_valid_name}
        train_df_location, valid_df_location = self.config.intermediate_train_valid_name.split("@")
        result[train_df_location] = train_df
        result[valid_df_location] = valid_df

        return {k: self.__apply_correct_names(df) for k, df in result.items()}

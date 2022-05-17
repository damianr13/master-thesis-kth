import glob
import os
from abc import ABC
from typing import Dict, TypeVar, Generic

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import WDCStandardizerConfig, BaseStandardizerConfig, \
    CSVNoSplitStandardizerConfig
from src.preprocess.definitions import BasePreprocessor

T = TypeVar("T", bound=BaseStandardizerConfig)


class BaseStandardizer(BasePreprocessor, Generic[T], ABC):
    def _sample_data(self, df: DataFrame) -> DataFrame:
        """
        Applies sampling to a dataframe according to the configured "train_sample_frac".

        This method is used for reducing the training sets to create situations where data is scarce, starting from the
        standard datasets.
        :param df:
        :return:
        """
        if self.config.train_sample_frac >= 1:
            return df

        left_ids = pd.Series(df['left_id'].unique()).sample(frac=self.config.train_sample_frac)
        weighted_df = df[df['left_id'].isin(left_ids)].copy()
        weights = weighted_df[['left_id', 'right_id']].groupby('right_id').count().rename(
            columns={'left_id': 'right_weight'})
        weighted_df = weighted_df.merge(weights, left_on='right_id', right_index=True)

        right_paired_ids = weighted_df.groupby('left_id', group_keys=False).apply(
            lambda s: s.sample(1, weights=weighted_df['right_weight']))['right_id'].unique()
        right_paired_ids = pd.Series(right_paired_ids)

        right_matching_ids = df[df['left_id'].isin(left_ids) & (df['label'] == 1)]['right_id'] \
            .unique()
        right_matching_ids = pd.Series(right_matching_ids)
        right_ids = right_matching_ids.sample(frac=self.config.train_sample_frac)
        right_ids = pd.concat([right_ids, right_paired_ids]).unique()
        right_ids = pd.Series(right_ids)

        right_all_ids = pd.Series(df['right_id'].unique())

        already_sampled_frac = len(right_ids) / len(right_all_ids)
        right_ids_pool = df[(df['left_id'].isin(left_ids))
                                       & (~df['right_id'].isin(right_ids))]['right_id'].unique()
        right_ids_pool = pd.Series(right_ids_pool)

        to_sample = (self.config.train_sample_frac - already_sampled_frac) * len(right_all_ids) / len(right_ids_pool)
        right_ids = pd.concat([right_ids, right_ids_pool.sample(frac=to_sample)])

        return df[df['left_id'].isin(left_ids) & df['right_id'].isin(right_ids)]

    def preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        df_for_location['train.csv'] = self._sample_data(df_for_location['train.csv'])
        return super().preprocess_all(df_for_location)


class RelationalDatasetStandardizer(BaseStandardizer[BaseStandardizerConfig]):
    """
    Applies standardization operation for datasets with a "relational format".

    These datasets have the 2 data sources in separate files, and 3 other files (train, test, valid) defining the
    train / test / valid split by referencing those 2 data sources. Instead, we need the 3 files to contain all the
    required data for working with the model
    """

    def __init__(self, config_path):
        super(RelationalDatasetStandardizer, self).__init__(config_path=config_path,
                                                            config_instantiator=BaseStandardizerConfig.parse_obj)

        self.source_a = pd.read_csv(os.path.join(self.config.original_location, 'tableA.csv'))
        self.source_b = pd.read_csv(os.path.join(self.config.original_location, 'tableB.csv'))

    def preprocess_one(self, refs_datas: DataFrame) -> DataFrame:
        result = refs_datas

        # merge the reference table with the data sources
        result = result.merge(self.source_a.add_prefix('left_'), left_on='ltable_id', right_on='left_id', how='inner')
        result = result.merge(self.source_b.add_prefix('right_'), left_on='rtable_id', right_on='right_id', how='inner')

        return super().preprocess_one(result)


class WDCDatasetStandardizer(BaseStandardizer[WDCStandardizerConfig]):
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

    def preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
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

        result = {k: self.__apply_correct_names(df) for k, df in result.items()}
        return super().preprocess_all(result)


class JSONLStandardizer(BaseStandardizer):
    """
    Standardizer for inputs in the jsonl format, split in different files. This is the format exported by a Spark job
    for example
    """
    def __init__(self, config_path):
        super(JSONLStandardizer, self).__init__(config_path=config_path,
                                                config_instantiator=BaseStandardizerConfig.parse_obj)

    @staticmethod
    def read_one_split_file(path) -> DataFrame:
        json_entries = glob.glob(f'{path}/*.json')

        first_json = json_entries[0]
        result = pd.read_json(first_json, lines=True)

        for entry in json_entries[1:]:
            print(f'Reading {entry}...')
            result = pd.concat([result, pd.read_json(entry, lines=True)])

        # a bit hardcoded for now, but let's see if this becomes an issue
        return result.rename(columns={'left_product_id': 'left_cluster_id', 'right_product_id': 'right_cluster_id'})


class CSVNoSplitStandardizer(BaseStandardizer[CSVNoSplitStandardizerConfig]):
    """
    Standardizer that reads plain csv files with the necessary information, but which are not split into
    train, validation and test sets.

    This standardizer performs the split according to the configured weights for each set
    """

    def __init__(self, config_path: str):
        super(CSVNoSplitStandardizer, self).__init__(config_path=config_path,
                                                     config_instantiator=CSVNoSplitStandardizerConfig.parse_obj)

    @staticmethod
    def __apply_specific_transformations(full_df: DataFrame) -> DataFrame:
        """
        Super specific stuff, may require changes for different csv datasets
        :param full_df:
        :return:
        """
        # this works because of the way the dataset is created.
        # We have an anchor product from one retailer (target), and we compare that with products from searches,
        # thus we can expect "target_name" to be common for more pairs, while "name" to be very specific so we can
        # assign individual ids to each offer on the right
        full_df['left_id'] = full_df.groupby('target_name').ngroup()
        full_df['right_id'] = np.arange(0, len(full_df))

        # change the label from string to integer
        full_df['label'] = full_df['label'].apply(lambda l: 1 if l == 'perfect' else 0)

        return full_df

    def preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        full_df = next(iter(df_for_location.values()))
        full_df = self.__apply_specific_transformations(full_df)

        splits = self.config.split_weights.values()
        # normalize splits
        splits = [s / sum(splits) for s in splits]
        # transform to cumulated and remove the last element (always 1)
        cumulated_splits = [sum(splits[0: i+1]) for i in range(len(splits) - 1)]
        # to absolute count
        cumulated_splits = [int(s * len(full_df)) for s in cumulated_splits]

        columns = full_df.columns
        split_list = np.split(full_df.sample(frac=1).values, cumulated_splits)
        target_names = self.config.split_weights.keys()
        result = dict(zip(target_names, split_list))
        result = {k: pd.DataFrame(v, columns=columns) for k, v in result.items()}

        return super().preprocess_all(result)

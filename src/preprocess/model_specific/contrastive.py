from itertools import chain
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import ContrastivePreprocessConfig
from src.preprocess.definitions import TransformerLMBasePreprocessor


class ContrastivePreprocessor(TransformerLMBasePreprocessor[ContrastivePreprocessConfig]):
    def __init__(self, config_path):
        super(ContrastivePreprocessor, self)\
            .__init__(config_path=config_path, config_instantiator=ContrastivePreprocessConfig.parse_obj)

    @staticmethod
    def _apply_pretrain_prefix(dataframe_name: str):
        return f'pretrain-{dataframe_name}'

    def extract_relevant_columns(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Override the parent method to exclude the pretraining set from columns selection.

        We do this because the column selection methods assumes there are "left_", respectively "right_" columns but,
        the pretraining already has the necessary columns extracted, and they refer to only one offer
        :param df_for_location:
        :return:
        """
        result = df_for_location.copy()
        pretrain_sets = {}
        for df in self.config.pretrain_used_datasets:
            pretrain_set_key = self._apply_pretrain_prefix(df)
            pretrain_sets[pretrain_set_key] = result.pop(pretrain_set_key)

        result = super().extract_relevant_columns(result)
        result.update(pretrain_sets)
        return result

    def extract_intermediately_relevant_columns(self, df) -> DataFrame:
        columns = ['left_id', 'right_id', 'left_text', 'right_text', 'label'] + \
            [c for c in self.config.non_textual_columns if c in df.columns] + \
            [f'left_{c}' for c in self.config.non_textual_columns if c not in df.columns] + \
            [f'right_{c}' for c in self.config.non_textual_columns if c not in df.columns]

        return df[columns]

    def _get_relevant_columns(self) -> List[str]:
        result = self.config.relevant_columns.copy()
        result.remove(ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME)
        if self.config.non_textual_columns is not None and len(self.config.non_textual_columns) > 0:
            result = list(set(result) - set(self.config.non_textual_columns))
        return result

    def preprocess_one(self, df: DataFrame) -> DataFrame:
        return self.extract_intermediately_relevant_columns(super().preprocess_one(df))

    def preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        return self.__separate_sources_pretrain(df_for_location)

    def __separate_sources_pretrain(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Retrieve the different offers in one dataframe. Assign a 'source' to each offer.

        This is useful for the contrastive pretraining where the goal is to make matching offers appear closer in the
        embedding space, while making non-matching offers more distant. To this end, each offer has to be encoded
        individually
        :param df_for_location:
        :return:
        """
        result = df_for_location.copy()
        for df_key in self.config.pretrain_used_datasets:
            df = df_for_location[df_key]
            result[self._apply_pretrain_prefix(df_key)] = self.__separate_sources_for_one(df).drop_duplicates()

        return result

    def separate_sources_for_one_as_tuple(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        columns_for_instance = ['text', ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME]
        left_instance_columns = {f'left_{c}': c for c in columns_for_instance}
        right_instance_columns = {f'right_{c}': c for c in columns_for_instance}

        # add non-textual columns as well
        # this assumes that non-textual columns are not in the for left_{c} or right_{c}
        left_instances_df = df[list(left_instance_columns.keys()) + self.config.non_textual_columns]\
            .rename(columns=left_instance_columns)
        right_instances_df = df[list(right_instance_columns.keys()) + self.config.non_textual_columns]\
            .rename(columns=right_instance_columns)

        left_instances_df = left_instances_df[
            left_instances_df[ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME].notnull()]
        left_instances_df = left_instances_df.drop_duplicates()
        right_instances_df = right_instances_df[
            right_instances_df[ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME].notnull()]
        right_instances_df = right_instances_df.drop_duplicates()

        return left_instances_df, right_instances_df

    @staticmethod
    def __perform_weighted_sampling(df: DataFrame, frac: float) -> DataFrame:
        # offers belonging to clusters should have a lower chance to be sampled
        # this way the head added in the 2nd phase learns how to accommodate offers outside the pre-training set
        df['weights'] = df.groupby('cluster_id')['cluster_id'].transform('count')
        df['weights'] = 1 / df['weights']
        df['weights'] = df['weights'] / df['weights'].sum()

        result = df.sample(frac=frac, weights=df['weights'])
        result.drop('weights', axis=1, inplace=True)

        return result

    def __separate_sources_for_one(self, df: DataFrame) -> DataFrame:
        left_instances_df, right_instances_df = self.separate_sources_for_one_as_tuple(df)
        if self.config.pretrain_right_sample is not None:
            right_instances_df = self.__perform_weighted_sampling(right_instances_df, self.config.pretrain_right_sample)

        result = pd.concat((left_instances_df, right_instances_df))
        result = self.__perform_weighted_sampling(result, self.config.pretrain_sample)

        return result


class ContrastivePreprocessorUnknownClusters(ContrastivePreprocessor):

    ITEM_ID_COLUMN_NAME = 'item_id'
    CLUSTER_ID_COLUMN_NAME = 'cluster_id'
    FULL_CLUSTER_COLUMN_NAME = 'full_cluster'

    def __init__(self, config_path: str):
        super(ContrastivePreprocessorUnknownClusters, self)\
            .__init__(config_path)

    @staticmethod
    def __get_unique_set_aggregator(val):
        if hasattr(val.iloc[0], '__iter__'):
            return tuple(set(chain(*val)))

        return tuple(set(val))

    def __extract_standalone_clusters(self, full_set: DataFrame, full_matches: DataFrame, id_column: str):
        possible_id_columns = [self.config.left_id_column, self.config.right_id_column]
        possible_id_columns.remove(id_column)
        other_id_column = possible_id_columns[0]

        result = full_set[~full_set[id_column].isin(full_matches[id_column])][[id_column]]
        result = result.groupby(by=id_column).aggregate({
            id_column: ContrastivePreprocessorUnknownClusters.__get_unique_set_aggregator
        })

        result[other_id_column] = [[]] * result.shape[0]
        return result

    def __extract_clusters(self, full_set: DataFrame) -> DataFrame:
        full_matches = full_set[full_set['label'] == 1]
        clusters_df = full_matches[[self.config.left_id_column, self.config.right_id_column]] \
            .groupby(by=self.config.left_id_column).agg({self.config.right_id_column: 'unique'}) \
            .reset_index()
        aux_col_name = 'aux'
        while True:
            """
            When the dataframe enters the loop it will have the structure: A, B 

            B is a set or a list (iterable), but A can be either a single id (for the first pass in the loop) or a set 
            (subsequent passes). 
            """
            expanded_frame = clusters_df.join(
                clusters_df.explode(self.config.right_id_column)[[self.config.right_id_column]].rename(
                    columns={self.config.right_id_column: aux_col_name}))
            contracted_frame = expanded_frame.groupby(aux_col_name).agg({
                self.config.left_id_column: ContrastivePreprocessorUnknownClusters.__get_unique_set_aggregator,
                self.config.right_id_column: ContrastivePreprocessorUnknownClusters.__get_unique_set_aggregator
            })

            if len(expanded_frame) == len(contracted_frame):
                break

            clusters_df = contracted_frame.reset_index(drop=True).groupby(self.config.right_id_column) \
                .agg({self.config.left_id_column: ContrastivePreprocessorUnknownClusters.__get_unique_set_aggregator}) \
                .reset_index()

        clusters_df = clusters_df[[self.config.left_id_column, self.config.right_id_column]]

        # add clusters of individual items
        standalone_left = self.__extract_standalone_clusters(full_set=full_set, full_matches=full_matches,
                                                             id_column=self.config.left_id_column)
        standalone_right = self.__extract_standalone_clusters(full_set=full_set, full_matches=full_matches,
                                                              id_column=self.config.right_id_column)
        clusters_df = pd.concat((clusters_df, standalone_left, standalone_right))
        clusters_df.insert(0,
                           ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME,
                           np.arange(0, len(clusters_df)))

        return clusters_df[[ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME,
                            self.config.left_id_column, self.config.right_id_column]]

    def __assign_clusters(self, target: DataFrame,
                          cluster_assignation_left: DataFrame,
                          cluster_assignation_right: DataFrame) -> DataFrame:
        """
        Assigns clusters to the offers.

        Here cluster is defined as a group of offers referring to the same product as deduced from the chain of positive
        pairs observed in the dataset.
        :param target:
        :param cluster_assignation_left:
        :param cluster_assignation_right:
        :return:
        """
        result = target.merge(cluster_assignation_left.rename(columns={
            ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME:
                f'left_{ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME}'
        }),
            left_on=self.config.left_id_column,
            right_on=ContrastivePreprocessorUnknownClusters.ITEM_ID_COLUMN_NAME,
            how='left')

        result = result.merge(cluster_assignation_right.rename(columns={
            ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME:
                f'right_{ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME}'
        }),
            left_on=self.config.right_id_column,
            right_on=ContrastivePreprocessorUnknownClusters.ITEM_ID_COLUMN_NAME,
            how='left')

        return result

    def separate_sources_for_one_as_tuple(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        For datasets where the clusters are inferred from labels, we provide source for performing source aware sampling
        :param df:
        :return:
        """
        left_instances_df, right_instances_df = super().separate_sources_for_one_as_tuple(df)

        left_instances_df['source'] = '#1'
        right_instances_df['source'] = '#2'
        return left_instances_df, right_instances_df

    def preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        full_set = pd.concat(list(df_for_location.values()))

        clusters_df = self.__extract_clusters(full_set=full_set)

        cluster_assignation_left = clusters_df.explode(self.config.left_id_column).rename(
            columns={self.config.left_id_column: ContrastivePreprocessorUnknownClusters.ITEM_ID_COLUMN_NAME})
        cluster_assignation_left = cluster_assignation_left[
            [ContrastivePreprocessorUnknownClusters.ITEM_ID_COLUMN_NAME,
             ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME]]

        cluster_assignation_right = clusters_df.explode(self.config.right_id_column).rename(
            columns={self.config.right_id_column: ContrastivePreprocessorUnknownClusters.ITEM_ID_COLUMN_NAME})
        cluster_assignation_right = cluster_assignation_right[
            [ContrastivePreprocessorUnknownClusters.ITEM_ID_COLUMN_NAME,
             ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME]]

        result = {k: self.__assign_clusters(v, cluster_assignation_left, cluster_assignation_right)
                  for k, v in df_for_location.items()}
        return super().preprocess_all(result)


class ContrastivePreprocessorKnownClusters(ContrastivePreprocessor):
    def extract_intermediately_relevant_columns(self, df) -> DataFrame:
        """
        Need to override this in order to include cluster columns
        :param df:
        :return:
        """
        return df[['left_cluster_id', 'right_cluster_id', 'left_text', 'right_text', 'label']]

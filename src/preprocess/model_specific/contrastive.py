from itertools import chain
from typing import Dict, Tuple

import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import ContrastivePreprocessConfig
from src.preprocess.definitions import BasePreprocessor


class ContrastivePreprocessor(BasePreprocessor[ContrastivePreprocessConfig]):
    def __init__(self, config_path):
        super(ContrastivePreprocessor, self)\
            .__init__(config_path=config_path, config_instantiator=ContrastivePreprocessConfig.parse_obj)

    def _extract_relevant_columns_one(self, df: DataFrame) -> DataFrame:
        return df[['left_text', 'right_text', 'label']]

    def _extract_relevant_columns(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Override the parent method to exclude the pretraining set from columns selection.

        We do this because the column selection methods assumes there are "left_", respectively "right_" columns but,
        the pretraining already has the necessary columns extracted, and they refer to only one offer
        :param df_for_location:
        :return:
        """
        result = df_for_location.copy()
        pretrain_set_df = result.pop(ContrastivePreprocessorUnknownClusters.PRETRAIN_SET_FILENAME)

        result = super()._extract_relevant_columns(result)
        result[ContrastivePreprocessorUnknownClusters.PRETRAIN_SET_FILENAME] = pretrain_set_df
        return result

    @staticmethod
    def extract_intermediately_relevant_columns(df) -> DataFrame:
        return df[['left_id', 'right_id', 'left_text', 'right_text', 'label']]

    def _preprocess_one(self, df: DataFrame) -> DataFrame:
        textual_columns = self.config.relevant_columns.copy()
        textual_columns.remove(ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME)

        left_textual_column = {f'left_{c}': c for c in textual_columns}
        right_textual_column = {f'right_{c}': c for c in textual_columns}

        def concat_with_annotation(sample, columns: Dict[str, str]):
            return ' '.join([f'[COL] {v} [VAL] {sample[k]}' for k, v in columns.items()])

        df['left_text'] = df.agg(lambda x: concat_with_annotation(x, left_textual_column), axis=1)
        df['right_text'] = df.agg(lambda x: concat_with_annotation(x, right_textual_column), axis=1)

        return self.extract_intermediately_relevant_columns(df)

    def _preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
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
        pretrain_set = None
        for df_key in self.config.pretrain_used_datasets:
            df = df_for_location[df_key]
            separated_df = self.__separate_sources_for_one(df)
            pretrain_set = separated_df if pretrain_set is None else pd.concat((pretrain_set, separated_df))

        pretrain_set = pretrain_set.drop_duplicates()
        result = df_for_location.copy()
        result[ContrastivePreprocessorUnknownClusters.PRETRAIN_SET_FILENAME] = pretrain_set
        return result

    def _separate_sources_for_one_as_tuple(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        df = df.sample(frac=self.config.pretrain_sample)

        columns_for_instance = ['text', ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME]
        left_instance_columns = {f'left_{c}': c for c in columns_for_instance}
        right_instance_columns = {f'right_{c}': c for c in columns_for_instance}

        left_instances_df = df[left_instance_columns.keys()].rename(columns=left_instance_columns)
        right_instances_df = df[right_instance_columns.keys()].rename(columns=right_instance_columns)

        left_instances_df = left_instances_df[
            left_instances_df[ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME].notnull()]
        left_instances_df = left_instances_df.drop_duplicates()
        right_instances_df = right_instances_df[
            right_instances_df[ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME].notnull()]
        right_instances_df = right_instances_df.drop_duplicates()

        return left_instances_df, right_instances_df

    def __separate_sources_for_one(self, df: DataFrame) -> DataFrame:
        left_instances_df, right_instances_df = self._separate_sources_for_one_as_tuple(df)
        return pd.concat((left_instances_df, right_instances_df))


class ContrastivePreprocessorUnknownClusters(ContrastivePreprocessor):

    ITEM_ID_COLUMN_NAME = 'item_id'
    CLUSTER_ID_COLUMN_NAME = 'cluster_id'
    FULL_CLUSTER_COLUMN_NAME = 'full_cluster'

    PRETRAIN_SET_FILENAME = 'pretrain.csv'

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
        clusters_df.insert(0, ContrastivePreprocessorUnknownClusters.CLUSTER_ID_COLUMN_NAME, range(0, len(clusters_df)))

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

    def _separate_sources_for_one_as_tuple(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        For datasets where the clusters are inferred from labels, we provide source for performing source aware sampling
        :param df:
        :return:
        """
        left_instances_df, right_instances_df = super()._separate_sources_for_one_as_tuple(df)

        left_instances_df['source'] = '#1'
        right_instances_df['source'] = '#2'
        return left_instances_df, right_instances_df

    def _preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
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
        return super()._preprocess_all(result)


class ContrastivePreprocessorKnownClusters(ContrastivePreprocessor):
    @staticmethod
    def extract_intermediately_relevant_columns(df) -> DataFrame:
        """
        Need to override this in order to include cluster columns
        :param df:
        :return:
        """
        return df[['left_cluster_id', 'right_cluster_id', 'left_id', 'right_id', 'left_text', 'right_text', 'label']]

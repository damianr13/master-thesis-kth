from itertools import chain
from typing import Dict

import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import ContrastivePreprocessConfig
from src.preprocess.definitions import BasePreprocessor


class ContrastivePreprocessor(BasePreprocessor[ContrastivePreprocessConfig]):

    ITEM_ID_COLUMN_NAME = 'item_id'
    CLUSTER_ID_COLUMN_NAME = 'cluster_id'
    FULL_CLUSTER_COLUMN_NAME = 'full_cluster'

    def __init__(self, config_path: str):
        super(ContrastivePreprocessor, self).__init__(config_path,
                                                      config_instantiator=ContrastivePreprocessConfig.parse_obj)

    @staticmethod
    def __get_unique_set_aggregator(val):
        if hasattr(val.iloc[0], '__iter__'):
            return tuple(set(chain(*val)))

        return tuple(set(val))

    def __extract_clusters(self, full_matches: DataFrame) -> DataFrame:
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
                self.config.left_id_column: ContrastivePreprocessor.__get_unique_set_aggregator,
                self.config.right_id_column: ContrastivePreprocessor.__get_unique_set_aggregator
            })

            if len(expanded_frame) == len(contracted_frame):
                break

            clusters_df = contracted_frame.reset_index(drop=True).groupby(self.config.right_id_column) \
                .agg({self.config.left_id_column: ContrastivePreprocessor.__get_unique_set_aggregator}) \
                .reset_index()

        clusters_df[ContrastivePreprocessor.FULL_CLUSTER_COLUMN_NAME] = \
            clusters_df[self.config.left_id_column] + clusters_df[self.config.right_id_column]
        clusters_df.insert(0, ContrastivePreprocessor.CLUSTER_ID_COLUMN_NAME, range(0, len(clusters_df)))

        return clusters_df[[ContrastivePreprocessor.CLUSTER_ID_COLUMN_NAME,
                            ContrastivePreprocessor.FULL_CLUSTER_COLUMN_NAME]]

    def __assign_clusters(self, target: DataFrame, cluster_assignation_df: DataFrame) -> DataFrame:
        result = target.merge(cluster_assignation_df.rename(columns={
            ContrastivePreprocessor.CLUSTER_ID_COLUMN_NAME: 'left_cluster'
        }),
            left_on=self.config.left_id_column,
            right_on=ContrastivePreprocessor.ITEM_ID_COLUMN_NAME)

        result = result.merge(cluster_assignation_df.rename(columns={
            ContrastivePreprocessor.CLUSTER_ID_COLUMN_NAME: 'right_cluster'
        }),
            left_on=self.config.right_id_column,
            right_on=ContrastivePreprocessor.ITEM_ID_COLUMN_NAME)

        return result

    def _preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        full_set = pd.concat(list(df_for_location.values()))
        full_matches = full_set[full_set['label'] == 1]

        clusters_df = self.__extract_clusters(full_matches=full_matches)
        cluster_assignation_df = clusters_df.explode(ContrastivePreprocessor.FULL_CLUSTER_COLUMN_NAME).rename(
            columns={ContrastivePreprocessor.FULL_CLUSTER_COLUMN_NAME: ContrastivePreprocessor.ITEM_ID_COLUMN_NAME})

        return {k: self.__assign_clusters(v, cluster_assignation_df) for k, v in df_for_location.items()}

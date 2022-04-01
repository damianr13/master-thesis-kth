from itertools import chain
from typing import Dict, List

import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import ContrastivePreprocessConfig
from src.preprocess.definitions import BasePreprocessor


class ContrastivePreprocessor(BasePreprocessor[ContrastivePreprocessConfig]):
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

        return clusters_df

    def _preprocess_all(self, df_for_location: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        full_set = pd.concat(list(df_for_location.values()))
        full_matches = full_set[full_set['label'] == 1]

        clusters_df = self.__extract_clusters(full_matches=full_matches)

        return super()._preprocess_all(df_for_location)

from typing import List

import pandas as pd
from pandas import DataFrame
import os
import json


class StandardizerConfig:
    """
    Defines the structure of a configurations file
    """
    def __init__(self, original_location: str, stand_location: str, relevant_columns: List[str]):
        self.original_location = original_location
        self.stand_location = stand_location
        self.relevant_columns = relevant_columns


class RelationalDatasetStandardizer:
    """
    Applies standardization operation for datasets with a "relational format".

    These datasets have the 2 data sources in separate files, and 3 other files (train, test, valid) defining the
    train / test / valid split by referencing those 2 data sources. Instead, we need the 3 files to contain all the
    required data for working with the model
    """
    def __init__(self, config_path):
        with open(config_path) as f:
            config_text = f.read()

        # load json config file as object
        self.config: StandardizerConfig = json.loads(config_text, object_hook=lambda d: StandardizerConfig(**d))

    def __standardize_one(self, refs_datas: DataFrame, source_a: DataFrame, source_b: DataFrame) -> DataFrame:
        result = refs_datas

        # merge the reference table with the data sources
        result = result.merge(source_a.add_prefix('left_'), left_on='ltable_id', right_on='left_id', how='inner')
        result = result.merge(source_b.add_prefix('right_'), left_on='rtable_id', right_on='right_id', how='inner')

        relevant_column_names = [f'left_{c}' for c in self.config.relevant_columns] + \
                                [f'right_{c}' for c in self.config.relevant_columns] + \
                                ['label']
        return result[relevant_column_names]

    def standardize(self):
        original_location = self.config.original_location
        stand_location = self.config.stand_location
        if not os.path.exists(stand_location):
            os.mkdir(stand_location)

        source_a = pd.read_csv(os.path.join(original_location, 'tableA.csv'))
        source_b = pd.read_csv(os.path.join(original_location, 'tableB.csv'))

        parts = ['train.csv', 'test.csv', 'valid.csv']
        for part in parts:
            part_refs = pd.read_csv(os.path.join(original_location, part))
            part_stand = self.__standardize_one(part_refs, source_a, source_b)

            part_stand.to_csv(os.path.join(stand_location, part))


def main():
    RelationalDatasetStandardizer(os.path.join('configs', 'stands_tasks', 'abt_buy.json')).standardize()
    RelationalDatasetStandardizer(os.path.join('configs', 'stands_tasks', 'amazon_google.json')).standardize()


if __name__ == "__main__":
    print(os.getcwd())
    main()

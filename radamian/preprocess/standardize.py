import os

import pandas as pd
from pandas import DataFrame

from radamian.preprocess.configs import StandardizerConfig
from radamian.preprocess.definitions import BasePreprocessor


class RelationalDatasetStandardizer(BasePreprocessor[StandardizerConfig]):
    """
    Applies standardization operation for datasets with a "relational format".

    These datasets have the 2 data sources in separate files, and 3 other files (train, test, valid) defining the
    train / test / valid split by referencing those 2 data sources. Instead, we need the 3 files to contain all the
    required data for working with the model
    """
    def __init__(self, config_path):
        super(RelationalDatasetStandardizer, self)\
            .__init__(config_path=config_path,
                      config_instantiator=lambda d: StandardizerConfig.parse_obj(d))

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


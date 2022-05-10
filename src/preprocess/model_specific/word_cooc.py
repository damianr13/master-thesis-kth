import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import ModelSpecificPreprocessConfig
from src.preprocess.definitions import NoColumnSelectionBasePreprocessor


class WordCoocPreprocessor(NoColumnSelectionBasePreprocessor):
    def __init__(self, config_path: str):
        super(WordCoocPreprocessor, self)\
            .__init__(config_path=config_path,
                      config_instantiator=lambda d: ModelSpecificPreprocessConfig.parse_obj(d))

    def preprocess_one(self, df: DataFrame) -> DataFrame:
        result = DataFrame()

        # concatenate all into one column, except for index and label
        left_columns = [f'left_{c}' for c in self.config.relevant_columns]
        right_columns = [f'right_{c}' for c in self.config.relevant_columns]

        result['ltext'] = pd.Series(df[left_columns].fillna('').values.tolist()).str.join(' ')
        result['rtext'] = pd.Series(df[right_columns].fillna('').values.tolist()).str.join(' ')
        result['label'] = df['label']

        return result

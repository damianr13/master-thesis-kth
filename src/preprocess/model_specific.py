import pandas as pd
from pandas import DataFrame

from src.preprocess.configs import ModelSpecificPreprocessConfig
from src.preprocess.definitions import BasePreprocessor


class WordCoocPreprocessor(BasePreprocessor):
    def __init__(self, config_path: str):
        super(WordCoocPreprocessor, self)\
            .__init__(config_path=config_path,
                      config_instantiator=lambda d: ModelSpecificPreprocessConfig.parse_obj(d))

    def _preprocess_one(self, df: DataFrame) -> DataFrame:
        result = DataFrame()

        # concatenate all into one column, except for index and label
        left_columns = [c for c in list(df) if c.startswith('left_')]
        right_columns = [c for c in list(df) if c.startswith('right_')]

        result['ltext'] = pd.Series(df[left_columns].fillna('').values.tolist()).str.join(' ')
        result['rtext'] = pd.Series(df[right_columns].fillna('').values.tolist()).str.join(' ')
        result['label'] = df['label']

        return result


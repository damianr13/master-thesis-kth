from typing import Dict

from src.preprocess.configs import BasePreprocConfig
from src.preprocess.definitions import TransformerLMBasePreprocessor


class DittoPreprocConfig(BasePreprocConfig):
    column_lengths: Dict[str, int]
    label_column: str


class DittoPreprocessor(TransformerLMBasePreprocessor[BasePreprocConfig]):
    def __init__(self, config_path: str):
        super(DittoPreprocessor, self).__init__(config_path=config_path,
                                                config_instantiator=DittoPreprocConfig.parse_obj)

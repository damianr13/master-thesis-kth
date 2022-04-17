from typing import List, Dict

from pydantic import BaseModel


class BasePreprocConfig(BaseModel):
    original_location: str
    target_location: str
    split_files: Dict[str, str]
    relevant_columns: List[str]


class WDCStandardizerConfig(BasePreprocConfig):
    train_valid_split_file: str
    intermediate_train_valid_name: str


class ModelSpecificPreprocessConfig(BasePreprocConfig):
    def __init__(self, **data):
        super(ModelSpecificPreprocessConfig, self).__init__(**data)


class ContrastivePreprocessConfig(BasePreprocConfig):
    left_id_column: str
    right_id_column: str
    label_column: str

    pretrain_used_datasets: List[str]
    pretrain_sample: float
    column_lengths: Dict[str, int]

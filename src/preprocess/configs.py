from typing import List, Dict, Any

from pydantic import BaseModel


class BasePreprocConfig(BaseModel):
    original_location: str
    target_location: str
    split_files: Dict[str, str]


class RelationalStandardizerConfig(BasePreprocConfig):
    """
    Defines the structure of a configurations file
    """
    relevant_columns: List[str]


class WDCStandardizerConfig(RelationalStandardizerConfig):
    train_valid_split_file: str
    intermediate_train_valid_name: str


class ModelSpecificPreprocessConfig(BasePreprocConfig):
    def __init__(self, *args, **kwargs):
        super(ModelSpecificPreprocessConfig, self).__init__(*args, **kwargs)

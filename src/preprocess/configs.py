from typing import List, Dict

from pydantic import BaseModel
from tap import Tap


class ExperimentsArgumentParser(Tap):
    no_train: bool = False
    debug: bool = False
    save_checkpoints: bool = False
    load_wandb_models: bool = False


class BasePreprocConfig(BaseModel):
    original_location: str = None
    target_location: str = None
    split_files: Dict[str, str]
    relevant_columns: List[str]


class BaseStandardizerConfig(BasePreprocConfig):
    train_sample_frac: float = 1


class WDCStandardizerConfig(BaseStandardizerConfig):
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

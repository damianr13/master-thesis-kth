from typing import List, Dict, Optional

from pydantic import BaseModel
from tap import Tap


class ExperimentsArgumentParser(Tap):
    no_train: bool = False
    debug: bool = False
    save_checkpoints: bool = False
    load_wandb_models: bool = False
    only_last_train: bool = False  # Report only the last round of training to wandb for a sweep

    learn_rate: Optional[float] = None
    batch_size: Optional[int] = None
    warmup_ratio: Optional[float] = None
    weight_decay: Optional[float] = None

    secondary_sequence: bool = False  # Whether we should launch the secondary 'test' objective of the program


class BasePreprocConfig(BaseModel):
    original_location: str = None
    target_location: str = None
    split_files: Dict[str, str]
    relevant_columns: List[str]
    rename_columns: Optional[Dict[str, str]]


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

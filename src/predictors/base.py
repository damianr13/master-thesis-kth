import os.path
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Tuple, Callable

import numpy as np
import pandas as pd
import torch.cuda
import wandb
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, IntervalStrategy, SchedulerType, PreTrainedTokenizer, \
    PreTrainedModel, AutoTokenizer, AutoModel, EvalPrediction
from wandb.apis.public import Run

from src import utils
from src.preprocess.configs import ExperimentsArgumentParser


class BasePredictor(ABC):
    name: str

    def __init__(self, name):
        self. name = name

    @abstractmethod
    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        pass

    @abstractmethod
    def test(self, test_set) -> float:
        pass


class DeepLearningHyperparameters(BaseModel):
    learning_rate: float = 5e-5
    epochs: int = 200
    batch_size: int = 64
    output: str = None
    loaders: int = 4
    parallel_batches: int = 1
    early_stop_patience: int = 10
    warmup_ratio: float = 0.05
    weight_decay: float = 0.00


class BaseTransformerClassifierConfig(BaseModel):
    transformer_name: str = 'distilbert-base-uncased'
    max_tokens: int = 256


class ClassificationDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx].copy()
        return example


class TransformerLMPredictor(BasePredictor, ABC):
    TRAIN_SEED = 97

    config: BaseTransformerClassifierConfig
    transformer: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    trainer: Trainer

    def __init__(self, name, report: bool = False, seed: int = 42):
        super(TransformerLMPredictor, self).__init__(name)
        self.report = report
        self.seed = seed

        self.fp16 = torch.cuda.is_available()

    def init_tokenizer_transformer(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_name,
                                                  additional_special_tokens=('[COL]', '[VAL]'))

        transformer: PreTrainedModel = AutoModel.from_pretrained(self.config.transformer_name)
        transformer.resize_token_embeddings(len(tokenizer))

        return tokenizer, transformer

    @staticmethod
    def init_default_output(hyperparameters: DeepLearningHyperparameters,
                            config_path: str,
                            suffix: str = '') -> DeepLearningHyperparameters:
        if hyperparameters.output is None:
            config_name = config_path.split('/')[-1][:-5]
            hyperparameters.output = os.path.join('output', config_name, suffix)

        return hyperparameters

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction):
        pred, labels = eval_pred
        pred = np.copy(pred)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        pred = pred.reshape(-1)
        labels = labels.reshape(-1)

        f1 = f1_score(labels, pred, pos_label=1, average='binary')
        return {'f1': f1}

    def perform_training(self, trainer: Trainer,
                         arguments: ExperimentsArgumentParser, output: str,
                         target: str,
                         evaluate: bool = False,
                         checkpoint_path: Optional[str] = None,
                         finish_run: bool = True, seed: int = 42):
        # if not checkpoint defined clearly and wandb loading is enabled check for a checkpoint there
        if arguments.load_wandb_models and not checkpoint_path:
            model_checkpoint = self._download_wandb_model(target=target, output=output)
            if model_checkpoint:
                self.load_trained(model_checkpoint)
                print(f'Successfully loaded model from path: {model_checkpoint}')
                return

        run = None
        if self.report:
            run_config = self.config.dict()
            output_split = output.split('/')
            run_config['current_target'] = output_split[-1]
            run_name = output_split[-2] + '_' + output_split[-1]
            run = wandb.init(project="master-thesis", entity="damianr13", config=run_config, name=run_name)

        if not checkpoint_path:
            utils.seed_all(seed)
            trainer.train()
        else:
            trainer.num_train_epochs = 50
            trainer.train(resume_from_checkpoint=checkpoint_path)

        if evaluate:
            trainer.evaluate(metric_key_prefix="eval")

        if output:
            trainer.save_model(output)

        if run and finish_run:
            run.finish()

        if not arguments.save_checkpoints:
            shutil.rmtree(output)

    @staticmethod
    def _download_wandb_model(target: str, output: str) -> Optional[str]:
        client = wandb.Api()

        previous_runs: Iterable[Run] = client.runs(path="damianr13/master-thesis", filters={
            "config.output_dir": output,
            "config.current_target": target
        })

        for run in previous_runs:
            artifacts: Iterable[wandb.Artifact] = run.logged_artifacts()
            for artifact in artifacts:
                if artifact.type != 'model':
                    continue

                artifact_dir = artifact.download()
                model_checkpoint = os.path.join(artifact_dir, 'pytorch_model.bin')
                if os.path.exists(model_checkpoint):
                    return model_checkpoint

        return None

    def _get_training_args(self, train_config: DeepLearningHyperparameters,
                           arguments: ExperimentsArgumentParser,
                           output: Optional[str] = None,
                           report: Optional[bool] = None) -> TrainingArguments:
        num_epochs = train_config.epochs if not arguments.debug else 1
        learning_rate = utils.select_first_available(
            [arguments.learn_rate, train_config.learning_rate])
        warmup_ratio = utils.select_first_available([arguments.warmup_ratio, train_config.warmup_ratio])
        batch_size = utils.select_first_available([arguments.batch_size, train_config.batch_size])
        weight_decay = utils.select_first_available([arguments.weight_decay, train_config.weight_decay])

        report = utils.select_first_available([report, self.report])
        report_to = "wandb" if report else "none"

        output = utils.select_first_available((output, train_config.output))

        return TrainingArguments(
            output_dir=output,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            max_grad_norm=1.0,
            weight_decay=weight_decay,
            seed=self.seed,
            fp16=self.fp16,
            save_steps=10,
            eval_steps=10,
            overwrite_output_dir=True,
            disable_tqdm=True,
            dataloader_num_workers=train_config.loaders,
            gradient_accumulation_steps=train_config.parallel_batches,
            report_to=[report_to],
            save_strategy=IntervalStrategy.EPOCH,
            lr_scheduler_type=SchedulerType.LINEAR,
            evaluation_strategy=IntervalStrategy.EPOCH,
            logging_strategy=IntervalStrategy.EPOCH,
            load_best_model_at_end=True
        )

    @abstractmethod
    def get_train_hyperparameters(self) -> DeepLearningHyperparameters:
        pass

    @abstractmethod
    def instantiate_classifier_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def instantiate_classifier_collator(self) -> any:
        pass

    @staticmethod
    def instantiate_classifier_dataset(df: pd.DataFrame) -> ClassificationDataset:
        return ClassificationDataset(df)

    def load_trained(self, checkpoint_path: Optional[str] = None, map_location: Optional[str] = None):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.get_train_hyperparameters().output, 'pytorch_model.bin')

        if map_location is None and not torch.cuda.is_available():
            map_location = torch.device('cpu')

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = self.instantiate_classifier_model()
        model.load_state_dict(checkpoint)

        # no sense of passing debug flag here since the model is only loaded and not trained
        collator = self.instantiate_classifier_collator()
        self.trainer = Trainer(model=model, data_collator=collator,
                               compute_metrics=self.compute_metrics)

    def test(self, test_set: DataFrame) -> float:
        test_dataset = self.instantiate_classifier_dataset(test_set)
        predict_results = self.trainer.predict(test_dataset)

        f1 = predict_results.metrics['test_f1']
        if wandb.run:
            wandb.log({'f1': f1})
        return f1


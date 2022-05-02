import copy
import os
import random
import shutil
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, cast

import numpy as np
import pandas as pd
import torch
import wandb
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.metrics import f1_score
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, PreTrainedTokenizerBase, \
    IntervalStrategy, SchedulerType, EarlyStoppingCallback
from wandb.apis.public import Run, Artifact

from src import utils
from src.augment import Augmenter
from src.preprocess.configs import ExperimentsArgumentParser
from src.predictors.base import BasePredictor


class DeepLearningHyperparameters(BaseModel):
    learning_rate: float = 5e-5
    epochs: int = 200
    batch_size: int = 64
    output: str = None
    loaders: int = 4
    parallel_batches: int = 1
    early_stop_patience: int = 10


class ContrastiveClassifierConfig(BaseModel):
    frozen: bool = False
    unfreeze: bool = False
    augment: str = "no"  # possible values: "no", "dropout", "mixda", "plain"
    swap_offers: bool = False
    dataset_name: str = 'unknown'
    transformer_name: str = 'distilbert-base-uncased'
    max_tokens: int = 128

    pretrain_specific: DeepLearningHyperparameters = DeepLearningHyperparameters()
    train_specific: DeepLearningHyperparameters = DeepLearningHyperparameters()


class ContrastivePretrainDataset(Dataset):

    def __init__(self, pretrain_df: DataFrame):
        self.data = pretrain_df

    def __getitem__(self, idx):
        item = self.data.iloc[idx].copy()
        positive = self.data[self.data['cluster_id'] == item['cluster_id']].sample(1).iloc[0].copy()

        return item, positive

    def __len__(self):
        return len(self.data)


class ContrastivePretrainDatasetWithSourceAwareSampling(Dataset):

    def __init__(self, pretrain_df: DataFrame):
        self.sources_map = self.__group_by_source(pretrain_df)
        self.source_list = list(self.sources_map.keys())

    @staticmethod
    def __group_by_source(df: DataFrame) -> Dict[str, DataFrame]:
        result = {}
        source_list = df['source'].drop_duplicates().tolist()
        for s in source_list:
            data_for_source = df[df['source'] == s]
            other_data = df[~df.isin(data_for_source)]

            corresponding_data = other_data[
                other_data['cluster_id'].isin(data_for_source['cluster_id'].drop_duplicates())]

            data_for_source = pd.concat((data_for_source, corresponding_data))
            result[s] = data_for_source

        max_data_length = max([len(data) for data in result.values()])
        result = {k: ContrastivePretrainDatasetWithSourceAwareSampling.__inflate_data(v, max_data_length)
                  for k, v in result.items()}

        return result

    @staticmethod
    def __inflate_data(df: DataFrame, target_length: int) -> DataFrame:
        current_length = len(df)
        if current_length >= target_length:
            return df

        diff = target_length - current_length
        allow_duplicates = diff > current_length
        inflating_set = df.sample(diff, replace=allow_duplicates)

        return pd.concat((df, inflating_set))

    @staticmethod
    def __extract_positives(df: DataFrame):
        return df[df['label'] == 1]

    def __len__(self):
        return len(self.sources_map[self.source_list[0]])

    def __getitem__(self, index) -> T_co:
        data_source_index = random.randint(0, len(self.source_list) - 1)
        data_source = self.sources_map[self.source_list[data_source_index]]

        item = data_source.iloc[index].copy()
        positive = data_source[data_source['cluster_id'] == item['cluster_id']].sample(1).iloc[0].copy()

        return item, positive


class ContrastiveClassificationDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx].copy()
        return example


@dataclass
class ContrastiveDataCollator(ABC):
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128

    def tokenize_features(self, features: List[str]):
        return self.tokenizer(features,
                              padding=True,
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")

    @staticmethod
    def collate_pair(batch_left, batch_right, labels):
        return {
            'input_ids_left': batch_left['input_ids'],
            'attention_mask_left': batch_left['attention_mask'],
            'labels': torch.LongTensor(labels),
            'input_ids_right': batch_right['input_ids'],
            'attention_mask_right': batch_right['attention_mask']
        }


@dataclass
class ContrastivePretrainingDataCollator(ContrastiveDataCollator):
    augment: str = "no"
    swap_offers: bool = False
    text_augmenter = Augmenter()

    def __call__(self, x):
        features_left = [v[0]['text'] for v in x]
        features_right = [v[1]['text'] for v in x]

        labels = [v[0]['cluster_id'] for v in x]

        if self.swap_offers:
            should_swap = random.choice([True, False])
            if should_swap:
                features_left, features_right = features_right, features_left

        if self.augment == 'plain':
            features_left = [self.text_augmenter.apply_aug(text) for text in features_left]
            features_right = [self.text_augmenter.apply_aug(text) for text in features_right]

        batch_left = self.tokenize_features(features_left)
        batch_right = self.tokenize_features(features_right)
        result = self.collate_pair(batch_left, batch_right, labels)

        if self.augment == 'mixda':
            aug_features_left = [self.text_augmenter.apply_aug(text) for text in features_left]
            aug_features_right = [self.text_augmenter.apply_aug(text) for text in features_right]

            aug_batch_left = self.tokenize_features(aug_features_left)
            aug_batch_right = self.tokenize_features(aug_features_right)

            aug_result = {f'aug_{k}': v for k, v in self.collate_pair(aug_batch_left, aug_batch_right, []).items()}
            aug_result.pop('aug_labels')
            result.update(aug_result)

        if self.augment == "dropout":
            # take advantage of dropout augmentation during training
            result = {k: torch.cat((v, v), dim=0) for k, v in result.items()}

        return result


@dataclass
class ContrastiveClassifierDataCollator(ContrastiveDataCollator):
    def __call__(self, x):
        features_left = [v['left_text'] for v in x]
        features_right = [v['right_text'] for v in x]

        labels = [v['label'] for v in x]
        batch_left = self.tokenize_features(features_left)
        batch_right = self.tokenize_features(features_right)

        return self.collate_pair(batch_left, batch_right, labels)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature

    def forward(self, features: Tensor, labels: Optional[torch.LongTensor] = None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * contrast_count).view(-1, 1).to(device), 0)

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        safe_exp_sum = torch.add(exp_logits.sum(1, keepdim=True), 1e-10)  # hack to avoid the infinity issue
        log_prob = logits - torch.log(safe_exp_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss


class AbstractContrastiveModel(nn.Module, ABC):
    transformer: nn.Module

    def apply_transformer(self, input_ids_left, attention_mask_left, input_ids_right, attention_mask_right):
        output_left = self.transformer(input_ids_left, attention_mask_left)
        output_right = self.transformer(input_ids_right, attention_mask_right)

        output_left = self.mean_pooling(output_left, attention_mask_left)
        output_right = self.mean_pooling(output_right, attention_mask_right)

        return output_left, output_right

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # output of the transformer encoder
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContrastivePretrainModel(AbstractContrastiveModel):
    def __init__(self, len_tokenizer: int, model: str, alpha_aug: float = 0.8):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

        self.criterion = SupConLoss()
        self.alpha_aug = alpha_aug

    def forward(self, input_ids_left, attention_mask_left, labels, input_ids_right, attention_mask_right,
                aug_input_ids_left=None, aug_attention_mask_left=None,
                aug_input_ids_right=None, aug_attention_mask_right=None):
        output_left, output_right = self.apply_transformer(input_ids_left, attention_mask_left,
                                                           input_ids_right, attention_mask_right)

        # MixDA
        if aug_input_ids_left and aug_attention_mask_left and aug_input_ids_right and aug_attention_mask_right:
            aug_output_left, aug_output_right = self.apply_transformer(aug_input_ids_left, aug_attention_mask_left,
                                                                       aug_input_ids_right, aug_attention_mask_right)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)

            output_left = output_left * aug_lam + aug_output_left * (1.0 - aug_lam)
            output_right = output_right * aug_lam + aug_output_right * (1.0 - aug_lam)

        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        return self.criterion(output, labels),


class ContrastiveClassifierHead(nn.Module):
    def __init__(self, hidden_size=1248, drop_out_probability: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(p=drop_out_probability),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.proj(x)


class ContrastiveClassifierModel(AbstractContrastiveModel):
    def __init__(self, len_tokenizer,
                 model: str,
                 checkpoint_path: Optional[str] = None,
                 existing_transformer: Optional[nn.Module] = None):
        super().__init__()

        if existing_transformer:
            self.transformer = existing_transformer
        else:
            self.transformer = AutoModel.from_pretrained(model)
            self.transformer.resize_token_embeddings(len_tokenizer)

        drop_out = self.transformer.config.hidden_dropout_prob \
            if hasattr(self.transformer.config, 'hidden_dropout_prob') else 0
        self.classification_head = ContrastiveClassifierHead(hidden_size=4 * self.transformer.config.hidden_size,
                                                             drop_out_probability=drop_out)
        self.criterion = nn.BCEWithLogitsLoss()

        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path), strict=False)

    def forward(self, input_ids_left, attention_mask_left, labels, input_ids_right, attention_mask_right):
        output_left, output_right = self.apply_transformer(input_ids_left, attention_mask_left,
                                                           input_ids_right, attention_mask_right)

        output = torch.cat((
            output_left, output_right,
            torch.abs(output_left - output_right),
            output_left * output_right), -1)

        projected = self.classification_head(output)

        loss = self.criterion(projected.view(-1), labels.float())
        # it applies sigmoid after loss, because we use "logits" in the loss function
        projected = torch.sigmoid(projected)

        return loss, projected


class ContrastivePredictor(BasePredictor):
    transformer: nn.Module = None
    trainer: Trainer

    def __init__(self, config_path: str, report: bool = False, seed: int = 42):
        super(ContrastivePredictor, self).__init__(name="contrastive")
        self.config: ContrastiveClassifierConfig = utils.load_as_object(
            config_path, ContrastiveClassifierConfig.parse_obj)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_name,
                                                       additional_special_tokens=('[COL]', '[VAL]'))

        self.report = report
        self.report_to = "wandb" if report else "none"
        self.seed = seed

        self._init_default_configs(config_path)

    def _init_default_configs(self, config_path: str):
        config_name = config_path.split('/')[-1][:-5]
        if self.config.train_specific.output is None:
            self.config.train_specific.output = os.path.join('output', config_name, 'train')

        if self.config.pretrain_specific.output is None:
            self.config.pretrain_specific.output = os.path.join('output', config_name, 'pretrain')

    @staticmethod
    def compute_metrics(eval_pred):
        pred, labels = eval_pred
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        pred = pred.reshape(-1)
        labels = labels.reshape(-1)

        f1 = f1_score(labels, pred, pos_label=1, average='binary')
        return {'f1': f1}

    def perform_training(self, trainer: Trainer, output: str, evaluate: bool = False,
                         checkpoint_path: Optional[str] = None,
                         finish_run: bool = True):
        run = None
        if self.report:
            run_config = self.config.dict()
            output_split = output.split('/')
            run_config['current_target'] = output_split[-1]
            run_name = output_split[-2] + '_' + output_split[-1]
            run = wandb.init(project="master-thesis", entity="damianr13", config=run_config, name=run_name)

        if not checkpoint_path:
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

    @staticmethod
    def _download_wandb_model(arguments: ExperimentsArgumentParser, target: str, output: str) -> Optional[str]:
        if not arguments.load_wandb_models:
            return None

        client = wandb.Api()
        previous_runs: Iterable[Run] = client.runs(path="damianr13/master-thesis", filters={
            f"config.{target}_specific.output": output,
            "config.current_target": target
        })

        for run in previous_runs:
            artifacts: Iterable[Artifact] = run.logged_artifacts()
            for artifact in artifacts:
                if artifact.type != 'model':
                    continue

                artifact_dir = artifact.download()
                model_checkpoint = os.path.join(artifact_dir, 'pytorch_model.bin')
                if os.path.exists(model_checkpoint):
                    return model_checkpoint

        return None

    def pretrain(self, pretrain_set: DataFrame, valid_set: DataFrame, arguments: ExperimentsArgumentParser,
                 source_aware_sampling: bool = True, checkpoint_path: Optional[str] = None) -> None:
        train_dataset = ContrastivePretrainDatasetWithSourceAwareSampling(pretrain_df=pretrain_set) \
            if source_aware_sampling else ContrastivePretrainDataset(pretrain_df=pretrain_set)
        valid_dataset = ContrastivePretrainDatasetWithSourceAwareSampling(pretrain_df=valid_set) \
            if source_aware_sampling else ContrastivePretrainDataset(pretrain_df=pretrain_set)

        model = ContrastivePretrainModel(len_tokenizer=len(self.tokenizer), model=self.config.transformer_name)

        num_epochs = self.config.pretrain_specific.epochs if not arguments.debug else 1

        training_args = TrainingArguments(output_dir=self.config.pretrain_specific.output,
                                          seed=self.seed,
                                          per_device_train_batch_size=self.config.pretrain_specific.batch_size,
                                          learning_rate=self.config.pretrain_specific.learning_rate,
                                          warmup_ratio=0.05,
                                          num_train_epochs=num_epochs,
                                          weight_decay=0.00,
                                          max_grad_norm=1.0,
                                          fp16=True,
                                          dataloader_num_workers=self.config.pretrain_specific.loaders,
                                          gradient_accumulation_steps=self.config.pretrain_specific.parallel_batches,
                                          metric_for_best_model="loss",
                                          disable_tqdm=True,
                                          report_to=[self.report_to],
                                          save_strategy=IntervalStrategy.EPOCH,
                                          save_steps=10,
                                          eval_steps=10,
                                          overwrite_output_dir=True,
                                          lr_scheduler_type=SchedulerType.LINEAR,
                                          logging_strategy=IntervalStrategy.EPOCH,
                                          load_best_model_at_end=True,
                                          evaluation_strategy=IntervalStrategy.EPOCH)

        collator = ContrastivePretrainingDataCollator(tokenizer=self.tokenizer,
                                                      max_length=self.config.max_tokens,
                                                      augment=self.config.augment,
                                                      swap_offers=self.config.swap_offers)
        trainer = Trainer(model=model,
                          train_dataset=train_dataset,
                          eval_dataset=valid_dataset,
                          args=training_args,
                          data_collator=collator)

        if self.config.pretrain_specific.early_stop_patience > 0:
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=self.config.pretrain_specific.early_stop_patience))

        existing_checkpoint = self._download_wandb_model(arguments,
                                                         target='pretrain',
                                                         output=self.config.pretrain_specific.output)
        if existing_checkpoint:
            self.load_pretrained(existing_checkpoint)
            print(f"Successfully loaded pretrained model: {existing_checkpoint}")
        else:
            report = self.report
            if arguments.only_last_train:
                self.report = False

            self.perform_training(trainer, output=self.config.pretrain_specific.output, checkpoint_path=checkpoint_path)

            self.report = report
            if not arguments.save_checkpoints:
                shutil.rmtree(self.config.pretrain_specific.output)

            self.transformer = model.transformer

        if self.config.frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def load_pretrained(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        model = ContrastivePretrainModel(checkpoint['transformer.embeddings.word_embeddings.weight'].shape[0],
                                         self.config.transformer_name)
        model.load_state_dict(checkpoint)

        self.transformer = model.transformer
        if self.config.frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def _init_training_trainer(self, model: ContrastiveClassifierModel,
                               train_dataset: ContrastiveClassificationDataset,
                               eval_dataset: ContrastiveClassificationDataset,
                               arguments: ExperimentsArgumentParser,
                               output: str,
                               allow_early_stop: bool = True,
                               report_overwrite: Optional[bool] = None):

        num_epochs = self.config.train_specific.epochs if not arguments.debug else 1
        learning_rate = utils.select_first_available(
            [arguments.learn_rate, self.config.train_specific.learning_rate])
        warmup_ratio = utils.select_first_available([arguments.warmup_ratio, 0.05])
        batch_size = utils.select_first_available([arguments.batch_size, self.config.train_specific.batch_size])
        weight_decay = utils.select_first_available([arguments.weight_decay, 0.01])

        report = utils.select_first_available([report_overwrite, self.report])
        report_to = "wandb" if report else "none"

        training_args = TrainingArguments(
            output_dir=output,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            max_grad_norm=1.0,
            weight_decay=weight_decay,
            seed=self.seed,
            fp16=True,
            save_steps=10,
            eval_steps=10,
            overwrite_output_dir=True,
            disable_tqdm=True,
            dataloader_num_workers=self.config.train_specific.loaders,
            gradient_accumulation_steps=self.config.train_specific.parallel_batches,
            report_to=[report_to],
            save_strategy=IntervalStrategy.EPOCH,
            lr_scheduler_type=SchedulerType.LINEAR,
            evaluation_strategy=IntervalStrategy.EPOCH,
            logging_strategy=IntervalStrategy.EPOCH,
            load_best_model_at_end=True
        )
        collator = ContrastiveClassifierDataCollator(tokenizer=self.tokenizer, max_length=self.config.max_tokens)

        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=collator,
                          compute_metrics=self.compute_metrics)

        if allow_early_stop and self.config.train_specific.early_stop_patience > 0:
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=self.config.train_specific.early_stop_patience))
        return trainer

    def load_trained(self, checkpoint_path: Optional[str] = None, map_location: Optional[str] = None):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.config.train_specific.output, 'pytorch_model.bin')

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = ContrastiveClassifierModel(checkpoint['transformer.embeddings.word_embeddings.weight'].shape[0],
                                           model=self.config.transformer_name)
        model.load_state_dict(checkpoint)

        collator = ContrastiveClassifierDataCollator(tokenizer=self.tokenizer, max_length=self.config.max_tokens)
        self.trainer = Trainer(model=model, data_collator=collator,
                               compute_metrics=self.compute_metrics)

    def train(self, train_set: DataFrame, valid_set: DataFrame,
              arguments: ExperimentsArgumentParser = ExperimentsArgumentParser()) -> None:
        train_dataset = ContrastiveClassificationDataset(df=train_set)
        eval_dataset = ContrastiveClassificationDataset(df=valid_set)
        model = ContrastiveClassifierModel(len_tokenizer=(len(self.tokenizer)),
                                           existing_transformer=self.transformer,
                                           model=self.config.transformer_name)

        # enforce config file settings for the first training
        arguments_copy = copy.copy(arguments)
        arguments_copy.learn_rate = None
        arguments_copy.warmup_ratio = None
        arguments_copy.weight_decay = None
        arguments_copy.batch_size = None

        report_overwrite = False if arguments.only_last_train else None
        trainer = self._init_training_trainer(model, train_dataset, eval_dataset, arguments_copy,
                                              self.config.train_specific.output, report_overwrite=report_overwrite)

        existing_checkpoint = self._download_wandb_model(arguments,
                                                         target='train',
                                                         output=self.config.train_specific.output)

        if existing_checkpoint:
            self.load_trained(existing_checkpoint)
            print(f"Successfully loaded trained model: {existing_checkpoint}")
        else:
            report = self.report
            if arguments.only_last_train and self.report:
                # prevent reporting for intermediary training
                self.report = False

            self.perform_training(trainer, output=self.config.train_specific.output, finish_run=True)

            self.report = report
            if not arguments.save_checkpoints:
                shutil.rmtree(self.config.train_specific.output)

            self.trainer = trainer

        # unfreeze the transformer after head has been initialized properly
        if self.config.frozen and self.config.unfreeze:
            if self.config.frozen:
                for param in self.transformer.parameters():
                    param.requires_grad = True

            output_train_2 = self.config.train_specific.output + "_2"
            model = cast(ContrastiveClassifierModel, self.trainer.model)
            trainer2 = self._init_training_trainer(model, train_dataset,
                                                   eval_dataset, arguments, output_train_2,
                                                   allow_early_stop=False)
            self.perform_training(trainer2, output=output_train_2, finish_run=False)

            self.trainer = trainer2

    def test(self, test_set: DataFrame) -> float:
        test_dataset = ContrastiveClassificationDataset(test_set)
        predict_results = self.trainer.predict(test_dataset)

        f1 = predict_results.metrics['test_f1']
        if wandb.run:
            wandb.log({'f1': f1})
        return f1

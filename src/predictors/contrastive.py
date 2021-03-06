import copy
import os
import random
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, cast, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoModel, Trainer, TrainingArguments, PreTrainedTokenizerBase, \
    IntervalStrategy, SchedulerType, EarlyStoppingCallback, PreTrainedModel

from src import utils
from src.augment import Augmenter
from src.performance.watcher import PerformanceWatcher, ExecutionSegment
from src.predictors.base import DeepLearningHyperparameters, TransformerLMPredictor, BaseTransformerClassifierConfig, \
    ClassificationDataset
from src.preprocess.configs import ExperimentsArgumentParser


class ContrastiveClassifierConfig(BaseTransformerClassifierConfig):
    frozen: bool = False
    unfreeze: bool = False
    augment: str = "no"  # possible values: "no", "dropout", "mixda", "plain"
    swap_offers: bool = False
    adaptive_tokenization: bool = False
    deepspeed: Optional[str] = None

    pretrain_specific: DeepLearningHyperparameters = DeepLearningHyperparameters()
    train_specific: DeepLearningHyperparameters = DeepLearningHyperparameters()
    train_2_specific: DeepLearningHyperparameters = DeepLearningHyperparameters()


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
        # TODO: this should not have been done here but in the collator
        data_source_index = random.randint(0, len(self.source_list) - 1)
        data_source = self.sources_map[self.source_list[data_source_index]]

        item = data_source.iloc[index].copy()
        positive = data_source[data_source['cluster_id'] == item['cluster_id']].sample(1).iloc[0].copy()

        return item, positive


@dataclass
class ContrastiveDataCollator(ABC):
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128
    debug: bool = False

    def tokenize_features(self, features: List[str]):
        if self.debug:
            PerformanceWatcher.get_instance().enter_segment(ExecutionSegment.TOKENIZING)

        result = self.tokenizer(features,
                                padding=True,
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")

        if self.debug:
            PerformanceWatcher.get_instance().exit_segment()

        return result

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
    transformer: PreTrainedModel
    debug: bool = False

    def apply_transformer(self, input_ids_left, attention_mask_left, input_ids_right, attention_mask_right):
        if self.debug:
            PerformanceWatcher.get_instance().enter_segment(ExecutionSegment.MODEL_FEEDFORWARD)

        output_left = self.transformer(input_ids_left, attention_mask_left)
        output_right = self.transformer(input_ids_right, attention_mask_right)

        output_left = self.mean_pooling(output_left, attention_mask_left)
        output_right = self.mean_pooling(output_right, attention_mask_right)

        if self.debug:
            PerformanceWatcher.get_instance().exit_segment()

        return output_left, output_right

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # output of the transformer encoder
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContrastivePretrainModel(AbstractContrastiveModel):
    def __init__(self, transformer: PreTrainedModel, alpha_aug: float = 0.8, debug: bool = False):
        super().__init__()
        self.transformer = transformer

        self.criterion = SupConLoss()
        self.alpha_aug = alpha_aug
        self.debug = debug

    def forward(self, input_ids_left, attention_mask_left, labels, input_ids_right, attention_mask_right,
                aug_input_ids_left=None, aug_attention_mask_left=None,
                aug_input_ids_right=None, aug_attention_mask_right=None):
        output_left, output_right = self.apply_transformer(input_ids_left, attention_mask_left,
                                                           input_ids_right, attention_mask_right)

        # MixDA
        if aug_input_ids_left is not None and aug_attention_mask_left is not None and \
                aug_input_ids_right is not None and aug_attention_mask_right is not None:
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
                 existing_transformer: Optional[PreTrainedModel] = None):
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


class ContrastivePredictor(TransformerLMPredictor):

    PRETRAIN_SEED = 13
    TRAIN_2_SEED = 23

    def __init__(self, config_path: str, report: bool = False, seed: int = 42):
        super(ContrastivePredictor, self).__init__(name="contrastive")
        self.config: ContrastiveClassifierConfig = utils.load_as_object(
            config_path, ContrastiveClassifierConfig.parse_obj)

        self.report = report
        self.report_to = "wandb" if report else "none"
        self.seed = seed

        self._init_default_configs(config_path)
        self.fp16 = torch.cuda.is_available()

        self.tokenizer, self.transformer = self.init_tokenizer_transformer()

    def _init_default_configs(self, config_path: str):
        self.config.pretrain_specific = self.init_default_output(self.config.pretrain_specific,
                                                                 config_path=config_path, suffix='pretrain')
        self.config.train_specific = self.init_default_output(self.config.train_specific,
                                                              config_path=config_path, suffix='train')

        if self.config.train_2_specific is not None:
            self.config.train_2_specific = self.init_default_output(self.config.train_2_specific,
                                                                    config_path=config_path, suffix='train_2')

    def perform_adaptive_tokenization(self, pretrain_set: DataFrame):
        tokens_df = pd.DataFrame()
        tokens_df['text_tokenized'] = pretrain_set['text'].apply(lambda x: self.tokenizer(x).tokens())

        def unite_tokens(tokens: List[str], index: int) -> List[Tuple[str, Tuple[str]]]:
            current_concat = tokens[index]
            i = index - 1
            result = []
            current_seed = [tokens[index]]
            while i > 0 and tokens[i + 1].startswith('##'):
                current_concat = tokens[i] + current_concat[2:]
                current_seed.insert(0, tokens[i])
                result.append((current_concat, tuple(current_seed)))
                i -= 1

            return result

        def extract_possible_concatenations(tokens: List[str]) -> List[str]:
            result = []
            for i in range(len(tokens) - 1):
                if tokens[i + 1].startswith('##'):
                    result += unite_tokens(tokens, i + 1)

            return result

        tokens_df['option'] = tokens_df['text_tokenized'].apply(extract_possible_concatenations)
        tokens_df = tokens_df.explode('option')
        tokens_df.dropna()
        tokens_df = tokens_df.groupby('option').option.count().rename('count').reset_index()

        present_at_least = len(pretrain_set) // 200
        tokens_df = tokens_df[tokens_df['count'] >= present_at_least]

        tokens_df['token'], tokens_df['seed'] = zip(*tokens_df['option'])
        all_seeds = tokens_df['seed'].tolist()

        clustered_seeds = {}  # keep the seed lists clustered by length
        for seed in all_seeds:
            cluster = len(seed)
            if cluster not in clustered_seeds:
                clustered_seeds[cluster] = []

            clustered_seeds[cluster].append(seed)

        useless_seeds = []  # these are the seeds for which a superset is also sampled
        for k in clustered_seeds.keys():
            if k + 1 not in clustered_seeds:
                continue

            for lower_rank_seed in clustered_seeds[k]:
                for higher_rank_seed in clustered_seeds[k+1]:
                    if set(lower_rank_seed).issubset(set(higher_rank_seed)):
                        useless_seeds.append(lower_rank_seed)

        tokens_df = tokens_df[~tokens_df['seed'].isin(useless_seeds)]
        self.tokenizer.add_tokens(tokens_df['token'].tolist(), special_tokens=True)

        vocab = self.tokenizer.get_vocab()
        tokens_df['seed_numeric'] = tokens_df['seed'].apply(lambda x: [vocab[t] for t in x])
        self.transformer.resize_token_embeddings(len(self.tokenizer))

        token_composition = dict(zip(tokens_df['token'].tolist(), tokens_df['seed_numeric'].tolist()))
        for token in token_composition.keys():
            token_average = torch.zeros((1, self.transformer.embeddings.word_embeddings.weight.data.shape[1]))
            for sub_token_id in token_composition[token]:
                token_average += self.transformer.embeddings.word_embeddings.weight.data[sub_token_id, :]

            token_average /= len(token_composition[token])
            self.transformer.embeddings.word_embeddings.weight.data[vocab[token], :] = token_average

    def pretrain(self, pretrain_set: DataFrame, valid_set: DataFrame, arguments: ExperimentsArgumentParser,
                 source_aware_sampling: bool = True, checkpoint_path: Optional[str] = None) -> None:
        train_dataset = self.instantiate_pretrain_dataset(pretrain_set, source_aware_sampling)
        valid_dataset = self.instantiate_pretrain_dataset(valid_set, source_aware_sampling)

        if self.config.adaptive_tokenization:
            self.perform_adaptive_tokenization(pretrain_set)

        model = ContrastivePretrainModel(transformer=self.transformer, debug=arguments.debug)

        num_epochs = self.config.pretrain_specific.epochs if not arguments.debug else 1

        training_args = TrainingArguments(output_dir=self.config.pretrain_specific.output,
                                          seed=self.seed,
                                          per_device_train_batch_size=self.config.pretrain_specific.batch_size,
                                          learning_rate=self.config.pretrain_specific.learning_rate,
                                          warmup_ratio=self.config.pretrain_specific.warmup_ratio,
                                          num_train_epochs=num_epochs,
                                          weight_decay=self.config.pretrain_specific.weight_decay,
                                          max_grad_norm=1.0,
                                          fp16=self.fp16,
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
                                          evaluation_strategy=IntervalStrategy.EPOCH,
                                          deepspeed=self.config.deepspeed)

        collator = self.instantiate_pretrain_collator(arguments=arguments)
        trainer = Trainer(model=model,
                          train_dataset=train_dataset,
                          eval_dataset=valid_dataset,
                          args=training_args,
                          data_collator=collator)

        report = self.report
        if arguments.only_last_train:
            self.report = False

        self.perform_training(trainer, arguments=arguments, target='pretrain',
                              output=self.config.pretrain_specific.output,
                              checkpoint_path=checkpoint_path, seed=self.PRETRAIN_SEED)

        self.report = report
        self.transformer = model.transformer

        if self.config.frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def perform_training(self, trainer: Trainer, arguments: ExperimentsArgumentParser, output: str, target: str,
                         evaluate: bool = False, checkpoint_path: Optional[str] = None, finish_run: bool = True,
                         seed: int = 42):
        # check if models from wandb should be loaded as pretrained models
        if arguments.load_wandb_models and not checkpoint_path and target == 'pretrain':
            model_checkpoint = self._download_wandb_model(target=target, output=output)
            if model_checkpoint:
                self.load_pretrained(model_checkpoint)
                print(f'Successfully loaded model from path: {model_checkpoint}')
                return

        # if we didn't load a pretrained model proceed with normal execution
        super().perform_training(trainer, arguments, output, target, evaluate, checkpoint_path, finish_run, seed)

    def load_pretrained(self, checkpoint_path: str, map_location: Optional[str] = None):
        if map_location is None and not torch.cuda.is_available():
            map_location = torch.device('cpu')

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = ContrastivePretrainModel(self.transformer)
        model.load_state_dict(checkpoint)

        self.transformer = model.transformer
        if self.config.frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def _init_training_trainer(self, model: ContrastiveClassifierModel,
                               train_dataset: ClassificationDataset,
                               eval_dataset: ClassificationDataset,
                               arguments: ExperimentsArgumentParser,
                               output: str,
                               train_config: DeepLearningHyperparameters,
                               allow_early_stop: bool = True,
                               report_overwrite: Optional[bool] = None):
        training_args = self._get_training_args(train_config, arguments, output, report_overwrite)

        collator = ContrastiveClassifierDataCollator(tokenizer=self.tokenizer,
                                                     max_length=self.config.max_tokens,
                                                     debug=arguments.debug)

        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=collator,
                          compute_metrics=self.compute_metrics)

        if allow_early_stop and train_config.early_stop_patience > 0:
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=train_config.early_stop_patience))
        return trainer

    def train(self, train_set: DataFrame, valid_set: DataFrame,
              arguments: ExperimentsArgumentParser = ExperimentsArgumentParser()) -> None:
        train_dataset = self.instantiate_classifier_dataset(df=train_set)
        eval_dataset = self.instantiate_classifier_dataset(df=valid_set)
        model = ContrastiveClassifierModel(len_tokenizer=(len(self.tokenizer)),
                                           existing_transformer=self.transformer,
                                           model=self.config.transformer_name)

        # enforce config file settings for the first training
        arguments_copy = copy.copy(arguments)
        if self.config.unfreeze:
            arguments_copy.learn_rate = None
            arguments_copy.warmup_ratio = None
            arguments_copy.weight_decay = None
            arguments_copy.batch_size = None

        report_overwrite = False if arguments.only_last_train else None
        trainer = self._init_training_trainer(model, train_dataset, eval_dataset, arguments_copy,
                                              self.config.train_specific.output,
                                              train_config=self.config.train_specific,
                                              report_overwrite=report_overwrite)
        report = self.report
        if arguments.only_last_train and self.report:
            # prevent reporting for intermediary training
            self.report = False

        # only finish the run if we have the "unfreeze" argument, meaning another training round follows
        self.perform_training(trainer, arguments=arguments_copy, target='train',
                              output=self.config.train_specific.output,
                              finish_run=self.config.unfreeze, seed=self.TRAIN_SEED)

        self.report = report

        # unfreeze the transformer after head has been initialized properly
        if self.config.frozen and self.config.unfreeze:
            if self.config.frozen:
                for param in self.transformer.parameters():
                    param.requires_grad = True

            output_train_2 = self.config.train_specific.output + "_2"
            model = cast(ContrastiveClassifierModel, self.trainer.model)
            train_config = self.config.train_2_specific if self.config.train_2_specific is not None \
                else self.config.train_specific
            trainer2 = self._init_training_trainer(model, train_dataset,
                                                   eval_dataset, arguments, output_train_2,
                                                   train_config=train_config,
                                                   allow_early_stop=False)
            self.perform_training(trainer2, arguments=arguments, output=output_train_2, target='train_2',
                                  finish_run=False, seed=self.TRAIN_2_SEED)

    def get_train_hyperparameters(self) -> DeepLearningHyperparameters:
        return self.config.train_specific

    def instantiate_classifier_model(self) -> torch.nn.Module:
        return ContrastiveClassifierModel(len(self.tokenizer), model=self.config.transformer_name)

    def instantiate_classifier_collator(self) -> any:
        return ContrastiveClassifierDataCollator(self.tokenizer, self.config.max_tokens)

    def instantiate_pretrain_collator(self, arguments: ExperimentsArgumentParser):
        return ContrastivePretrainingDataCollator(tokenizer=self.tokenizer,
                                                  max_length=self.config.max_tokens,
                                                  augment=self.config.augment,
                                                  swap_offers=self.config.swap_offers,
                                                  debug=arguments.debug)

    @staticmethod
    def instantiate_pretrain_dataset(df: DataFrame, source_aware_sampling: bool = True):
        return ContrastivePretrainDatasetWithSourceAwareSampling(pretrain_df=df) \
            if source_aware_sampling else ContrastivePretrainDataset(pretrain_df=df)


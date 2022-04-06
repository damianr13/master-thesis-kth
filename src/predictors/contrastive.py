import random
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import f1_score
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, PreTrainedTokenizerBase

from src.predictors.base import BasePredictor


class ContrastivePretrainDataset(Dataset):

    def __init__(self, pretrain_df: DataFrame,
                 tokenizer_identifier: str = 'huawei-noah/TinyBERT_General_4L_312D',
                 max_length=128):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier,
                                                       additional_special_tokens=('[COL]', '[VAL]'))

        self.sources_map = ContrastivePretrainDataset.__group_by_source(pretrain_df)
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
        result = {k: ContrastivePretrainDataset.__inflate_data(v, max_data_length) for k, v in result.items()}

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
    def __init__(self, df: DataFrame, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx].copy()
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
    def __call__(self, x):
        features_left = [v[0]['text'] for v in x]
        features_right = [v[1]['text'] for v in x]

        labels = [v[0]['cluster_id'] for v in x]
        batch_left = self.tokenize_features(features_left)
        batch_right = self.tokenize_features(features_right)

        return self.collate_pair(batch_left, batch_right, labels)


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

    def forward(self, features: Tensor, labels: Optional[torch.LongTensor] = None, mask=None):
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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

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


class ContrastivePretrainModel(AbstractContrastiveModel):
    def __init__(self, len_tokenizer: int, model: str = 'huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

        self.criterion = SupConLoss()

    def forward(self, input_ids_left, attention_mask_left, labels, input_ids_right, attention_mask_right):
        output_left, output_right = self.apply_transformer(input_ids_left, attention_mask_left,
                                                           input_ids_right, attention_mask_right)

        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        return self.criterion(output, labels)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # output of the transformer encoder
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContrastiveClassifierHead(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.proj(x)


class ContrastiveClassifierModel(AbstractContrastiveModel):
    def __init__(self, len_tokenizer,
                 checkpoint_path: Optional[str] = None,
                 existing_transformer: Optional[nn.Module] = None,
                 model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()

        if existing_transformer:
            self.transformer = existing_transformer
        else:
            self.transformer = AutoModel.from_pretrained(model)
            self.transformer.resize_token_embeddings(len_tokenizer)

        self.classification_head = ContrastiveClassifierHead()
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
        projected = torch.sigmoid(projected)  # TODO: understand why we modify the projections after computing loss

        return loss, projected


class ContrastivePredictor(BasePredictor):
    transformer: nn.Module
    trainer: Trainer

    @staticmethod
    def compute_metrics(eval_pred):
        pred, labels = eval_pred
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        pred = pred.reshape(-1)
        labels = labels.reshape(-1)

        f1 = f1_score(labels, pred, pos_label=1, average='binary')
        return {'f1': f1}

    def pretrain(self, pretrain_set: DataFrame) -> None:
        train_dataset = ContrastivePretrainDataset(pretrain_df=pretrain_set)
        model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer))
        training_args = TrainingArguments(output_dir='output',
                                          per_device_train_batch_size=4,
                                          learning_rate=5e-05,
                                          warmup_ratio=0.05,
                                          num_train_epochs=200,
                                          weight_decay=0.01,
                                          max_grad_norm=1.0,
                                          # fp16=True,
                                          dataloader_num_workers=4,
                                          disable_tqdm=True,
                                          report_to=["none"],
                                          lr_scheduler_type="linear",
                                          save_strategy="epoch",
                                          logging_strategy="epoch")
        collator = ContrastivePretrainingDataCollator(tokenizer=train_dataset.tokenizer)

        trainer = Trainer(model=model,
                          train_dataset=train_dataset,
                          args=training_args,
                          data_collator=collator,
                          compute_metrics=self.compute_metrics)

        trainer.train()
        self.trainer = trainer

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        train_dataset = ContrastiveClassificationDataset(df=train_set)
        eval_dataset = ContrastiveClassificationDataset(df=valid_set)
        model = ContrastiveClassifierModel(len_tokenizer=(len(train_dataset.tokenizer)),
                                           existing_transformer=self.transformer)
        training_args = TrainingArguments(
            output_dir='output',
            per_device_train_batch_size=4,
            learning_rate=1e-3,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            weight_decay=0.01,
            seed=42,
            dataloader_num_workers=4,
            disable_tqdm=True,
            report_to=["none"],
            lr_scheduler_type="linear",
            save_strategy="epoch",
            logging_strategy="epoch"
        )
        collator = ContrastiveClassifierDataCollator(tokenizer=train_dataset.tokenizer)

        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=collator,
                          compute_metrics=self.compute_metrics)
        trainer.train()
        trainer.evaluate(metric_key_prefix="eval")
        self.trainer = trainer

    def test(self, test_set: DataFrame) -> float:
        test_dataset = ContrastiveClassificationDataset(test_set)
        predict_results = self.trainer.predict(test_dataset)

        return predict_results.metrics['f1']


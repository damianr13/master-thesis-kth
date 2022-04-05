import random
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import torch.nn
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, AutoModel

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
        data_source_index = random.randint(0, len(self.source_list))
        data_source = self.sources_map[self.source_list[data_source_index]]

        item = data_source.iloc[index].copy()
        positive = data_source[data_source['cluster_id'] == item['cluster_id']].sample(1).iloc[0].copy()

        return item, positive


@dataclass
class ContrastiveDataCollator:
    def __call__(self, x):
        pass


class ContrastivePretrainModel(torch.nn.Module):
    def __init__(self, len_tokenizer: int, model: str = 'huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

        self.criterion = None

    def forward(self, input_ids_left, attention_mask_left, labels, input_ids_right, attention_mask_right):
        output_left = self.transformer(input_ids_left, attention_mask_left)
        output_right = self.transformer(input_ids_right, attention_mask_right)

        output_left = self.mean_pooling(output_left, attention_mask_left)
        output_right = self.mean_pooling(output_right, attention_mask_right)

        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        return self.criterion(output, labels)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # output of the transformer encoder
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContrastivePredictor(BasePredictor):
    def pretrain(self, pretrain_set: DataFrame) -> None:
        pretrain_dataset = None

        pass

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        pass

    def test(self, test_set) -> float:
        pass

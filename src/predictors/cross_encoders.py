import os
from dataclasses import dataclass

import torch
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase, PreTrainedModel, Trainer

from src import utils
from src.predictors.base import BaseTransformerClassifierConfig, TransformerLMPredictor
from src.predictors.base import DeepLearningHyperparameters
from src.preprocess.configs import ExperimentsArgumentParser


class DittoClassifierConfig(BaseTransformerClassifierConfig):
    hyperparameters: DeepLearningHyperparameters = DeepLearningHyperparameters()


@dataclass
class DittoDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256

    def __call__(self, x):
        features = [(v['left_text'], v['right_text']) for v in x]

        labels = [v['label'] for v in x]

        batch = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=features,
                                                 max_length=self.max_length,
                                                 padding=True,
                                                 truncation=True,
                                                 return_tensors='pt')

        batch['labels'] = torch.FloatTensor(labels)
        return batch


class DittoModel(torch.nn.Module):
    transformer: PreTrainedModel

    def __init__(self, transformer: PreTrainedModel):
        super().__init__()

        self.transformer = transformer

        self.head = torch.nn.Linear(self.transformer.config.hidden_size, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        enc = self.transformer(input_ids, attention_mask, token_type_ids)[0][:, 0, :]  # CLS token

        prediction = self.head(enc)
        loss = self.criterion(prediction.view(-1), labels.float()) if labels is not None else None

        prediction = torch.sigmoid(prediction)
        return loss, prediction


class DittoPredictor(TransformerLMPredictor):
    def __init__(self, config_path: str, report: bool = False, seed: int = 42):
        super(DittoPredictor, self).__init__(name='ditto')
        self.config: DittoClassifierConfig = utils.load_as_object(config_path, DittoClassifierConfig.parse_obj)
        if self.config.hyperparameters.output is None:
            config_name = config_path.split('/')[-1][:-5]
            self.config.hyperparameters.output = os.path.join('output', config_name)

        self.report = report
        self.seed = seed

        self.fp16 = torch.cuda.is_available()

        self.tokenizer, self.transformer = self.init_tokenizer_transformer()

    def train(self, train_set: DataFrame, valid_set: DataFrame,
              arguments: ExperimentsArgumentParser = ExperimentsArgumentParser()) -> None:
        train_dataset = self.instantiate_classifier_dataset(train_set)
        valid_dataset = self.instantiate_classifier_dataset(valid_set)

        collator = self.instantiate_classifier_collator()

        model = self.instantiate_classifier_model()
        training_args = self._get_training_args(self.config.hyperparameters, arguments=arguments)
        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=valid_dataset,
                          data_collator=collator,
                          compute_metrics=self.compute_metrics)

        self.perform_training(trainer, arguments=arguments,
                              output=self.config.hyperparameters.output,
                              finish_run=False, seed=self.TRAIN_SEED)

    def get_train_hyperparameters(self) -> DeepLearningHyperparameters:
        return self.config.hyperparameters

    def instantiate_classifier_model(self) -> DittoModel:
        return DittoModel(self.transformer)

    def instantiate_classifier_collator(self) -> DittoDataCollator:
        return DittoDataCollator(self.tokenizer, self.config.max_tokens)

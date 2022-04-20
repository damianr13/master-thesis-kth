import os
import random
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import transformers
import wandb

from src.predictors.base import BasePredictor
from src.predictors.contrastive import ContrastivePredictor
from src.predictors.dummy import AllMatchPredictor, NoMatchPredictor, BalancedPredictor, ClassDistributionAwarePredictor
from src.predictors.word_cooc import WordCoocPredictor
from src.preprocess.definitions import BasePreprocessor
from src.preprocess.model_specific.contrastive import ContrastivePreprocessorKnownClusters
from src.preprocess.model_specific.word_cooc import WordCoocPreprocessor
from src.preprocess.standardize import RelationalDatasetStandardizer, WDCDatasetStandardizer


def run_pipeline(stand_config: str, preproc_config: str, predictor: BasePredictor,
                 standardizer_init: Callable[[str], BasePreprocessor] = RelationalDatasetStandardizer,
                 preprocessor_init: Callable[[str], BasePreprocessor] = WordCoocPreprocessor) -> Tuple[str, float]:
    standardizer = standardizer_init(stand_config)
    preprocessor = preprocessor_init(preproc_config)

    standardizer.preprocess()
    preprocessor.preprocess()
    train_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'test.csv'))

    predictor.train(train_df, valid_df)
    return predictor.name, predictor.test(test_df)


def run_experiments_for_predictor(predictor: BasePredictor,
                                  results_table: wandb.Table,
                                  preproc_for_model: Optional[str] = None) -> None:
    if not preproc_for_model:
        preproc_for_model = predictor.name

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'abt_buy.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'abt_buy.json'),
                                  predictor=predictor)
    results_table.add_data('abt_buy', model_name, f1)

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'amazon_google.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'amazon_google.json'),
                                  predictor=predictor)
    results_table.add_data('amazon_google', model_name, f1)

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'wdc_computers_large.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'wdc_computers_large.json'),
                                  predictor=predictor,
                                  standardizer_init=WDCDatasetStandardizer)
    results_table.add_data('wdc_computers_large', model_name, f1)


def main():
    wandb.init(project="master-thesis", entity="damianr13")
    f1_table = wandb.Table(columns=['experiment', 'model', 'score'])

    run_experiments_for_predictor(
        predictor=WordCoocPredictor(os.path.join('configs', 'model_train', 'word_cooc.json')),
        results_table=f1_table)

    run_experiments_for_predictor(predictor=AllMatchPredictor(), preproc_for_model='word_cooc', results_table=f1_table)

    run_experiments_for_predictor(predictor=NoMatchPredictor(), preproc_for_model='word_cooc', results_table=f1_table)

    run_experiments_for_predictor(predictor=BalancedPredictor(), preproc_for_model='word_cooc', results_table=f1_table)

    run_experiments_for_predictor(predictor=ClassDistributionAwarePredictor(),
                                  preproc_for_model='word_cooc',
                                  results_table=f1_table)

    wandb.log({"f1_scores": f1_table})


def stuff():
    WDCDatasetStandardizer(os.path.join('configs', 'stands_tasks', 'wdc_computers_medium.json')).preprocess()
    ContrastivePreprocessorKnownClusters(
        os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json')).preprocess()
    #
    pretrain_train_set = pd.read_csv(os.path.join('data', 'processed', 'contrastive', 'wdc_computers_medium',
                                                  'pretrain-train.csv'))
    pretrain_valid_set = pd.read_csv(os.path.join('data', 'processed', 'contrastive', 'wdc_computers_medium',
                                                  'pretrain-valid.csv'))
    train_set = pd.read_csv(os.path.join('data', 'processed', 'contrastive', 'wdc_computers_medium', 'train.csv'))
    valid_set = pd.read_csv(os.path.join('data', 'processed', 'contrastive', 'wdc_computers_medium', 'valid.csv'))
    test_set = pd.read_csv(os.path.join('data', 'processed', 'contrastive', 'wdc_computers_medium', 'test.csv'))
    #
    predictor = ContrastivePredictor(config_path=os.path.join('configs', 'model_train', 'contrastive',
                                                              'frozen_no-aug_wdc-computers-medium.json'),
                                     report=True, seed=42)
    predictor.pretrain(pretrain_set=pretrain_train_set, valid_set=pretrain_valid_set, source_aware_sampling=False)
    predictor.train(train_set, valid_set)
    print("Trained")
    f1 = predictor.test(test_set)
    #
    print(f'Finished with resulting f1 {f1}')


def seed_all(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)


if __name__ == "__main__":
    print(os.getcwd())
    seed_all(42)
    stuff()

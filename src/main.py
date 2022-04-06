import os
from typing import Callable, Tuple, Optional

import pandas as pd
import wandb

from src.predictors.base import BasePredictor
from src.predictors.contrastive import ContrastivePretrainDataset, ContrastivePredictor
from src.predictors.dummy import AllMatchPredictor, NoMatchPredictor, BalancedPredictor, ClassDistributionAwarePredictor
from src.predictors.word_cooc import WordCoocPredictor
from src.preprocess.definitions import BasePreprocessor
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
    pretrain_set = pd.read_csv(os.path.join('data', 'processed', 'contrastive', 'abt_buy', 'pretrain.csv'))

    predictor = ContrastivePredictor(name='contrastive')
    predictor.pretrain(pretrain_set)

    print('Finished')


if __name__ == "__main__":
    print(os.getcwd())
    stuff()

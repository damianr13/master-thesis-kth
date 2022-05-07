import os
import random
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from pydantic import BaseModel

from src.predictors.base import BasePredictor
from src.predictors.contrastive import ContrastivePredictor
from src.predictors.dummy import AllMatchPredictor, NoMatchPredictor, BalancedPredictor, ClassDistributionAwarePredictor
from src.predictors.word_cooc import WordCoocPredictor
from src.preprocess.configs import ExperimentsArgumentParser
from src.preprocess.definitions import BasePreprocessor
from src.preprocess.model_specific.contrastive import ContrastivePreprocessorKnownClusters, \
    ContrastivePreprocessorUnknownClusters
from src.preprocess.model_specific.word_cooc import WordCoocPreprocessor
from src.preprocess.standardize import RelationalDatasetStandardizer, WDCDatasetStandardizer
from src.utils import seed_all


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


def run_baseline_experiments():
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


class SupConExperimentConfig(BaseModel):
    stand_path: str
    proc_path: str
    predictor_path: str
    known_clusters: bool


def run_single_supcon_experiment(experiment_config: SupConExperimentConfig,
                                 arguments: ExperimentsArgumentParser):
    known_clusters = experiment_config.known_clusters
    standardizer = WDCDatasetStandardizer(experiment_config.stand_path) if known_clusters \
        else RelationalDatasetStandardizer(experiment_config.stand_path)
    standardizer.preprocess()
    preprocessor = ContrastivePreprocessorKnownClusters(experiment_config.proc_path) if known_clusters \
        else ContrastivePreprocessorUnknownClusters(experiment_config.proc_path)

    target_split = standardizer.config.target_location.split('/')
    directory = target_split[-1] if target_split[-1] != '' else target_split[-2]
    default_preproc_target = os.path.join('data', 'processed', 'contrastive', directory)

    preprocessor.preprocess(original_location=standardizer.config.target_location,
                            target_location=default_preproc_target)

    if arguments.no_train:
        return

    pretrain_train_set = pd.read_csv(os.path.join(default_preproc_target, 'pretrain-train.csv'))
    pretrain_valid_set = pd.read_csv(os.path.join(default_preproc_target, 'pretrain-valid.csv'))
    train_set = pd.read_csv(os.path.join(default_preproc_target, 'train.csv'))
    valid_set = pd.read_csv(os.path.join(default_preproc_target, 'valid.csv'))
    test_set = pd.read_csv(os.path.join(default_preproc_target, 'test.csv'))

    predictor = ContrastivePredictor(config_path=experiment_config.predictor_path, report=not arguments.debug, seed=42)
    predictor.pretrain(pretrain_set=pretrain_train_set, valid_set=pretrain_valid_set,
                       source_aware_sampling=not known_clusters, arguments=arguments)
    predictor.train(train_set, valid_set, arguments=arguments)
    print("Trained")
    f1 = predictor.test(test_set)

    print(f'Finished with resulting f1 {f1}')

    if wandb.run:
        wandb.run.finish()


def run_supcon_experiments(arguments: ExperimentsArgumentParser):
    experiments = [
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample50_wdc-computers-medium.json'),
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled'
                                           'frozen_no-aug_batch-pt128_adaptive-tokenization_sample50_'
                                                                                    '_wdc-computers-medium.json'),
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_aug-mixda_swap_batch-pt64_sample50_wdc-computers-medium.json'),
            "known_clusters": True
        }
    ]

    for exp in experiments:
        experiment_config = SupConExperimentConfig.parse_obj(exp)
        run_single_supcon_experiment(experiment_config, arguments)


if __name__ == "__main__":
    args = ExperimentsArgumentParser().parse_args()

    print(os.getcwd())
    seed_all(42)
    torch.cuda.seed_all()
    run_supcon_experiments(args)

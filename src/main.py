import os
from datetime import datetime
from typing import Callable, Tuple, Optional, List, Dict

import pandas as pd
import torch
import wandb
from pydantic import BaseModel

from src.performance.watcher import PerformanceWatcher
from src.predictors.base import BasePredictor
from src.predictors.contrastive import ContrastivePredictor
from src.predictors.cross_encoders import DittoPredictor
from src.predictors.dummy import AllMatchPredictor, NoMatchPredictor, BalancedPredictor, ClassDistributionAwarePredictor
from src.predictors.word_cooc import WordCoocPredictor
from src.preprocess.configs import ExperimentsArgumentParser
from src.preprocess.definitions import BasePreprocessor
from src.preprocess.model_specific.contrastive import ContrastivePreprocessorKnownClusters, \
    ContrastivePreprocessorUnknownClusters
from src.preprocess.model_specific.ditto import DittoPreprocessor
from src.preprocess.model_specific.word_cooc import WordCoocPreprocessor
from src.preprocess.standardize import RelationalDatasetStandardizer, WDCDatasetStandardizer, JSONLStandardizer, \
    BaseStandardizer, CSVNoSplitStandardizer
from src.utils import seed_all


def run_pipeline(stand_config: str, preproc_config: str, predictor: BasePredictor,
                 standardizer_init: Callable[[str], BasePreprocessor] = RelationalDatasetStandardizer,
                 preprocessor_init: Callable[[str], BasePreprocessor] = WordCoocPreprocessor,
                 arguments: ExperimentsArgumentParser = ExperimentsArgumentParser()) -> Tuple[str, float]:
    standardizer = standardizer_init(stand_config)
    preprocessor = preprocessor_init(preproc_config)

    standardizer.preprocess()
    preprocessor.preprocess()
    train_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'test.csv'))

    if arguments.no_train:
        return predictor.name, 0

    predictor.train(train_df, valid_df)
    return predictor.name, predictor.test(test_df)


def run_experiments_for_predictor(predictor: BasePredictor,
                                  results_table: wandb.Table,
                                  arguments: ExperimentsArgumentParser,
                                  preproc_for_model: Optional[str] = None) -> None:
    if not preproc_for_model:
        preproc_for_model = predictor.name

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'abt_buy.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'abt_buy.json'),
                                  predictor=predictor,
                                  arguments=arguments)
    results_table.add_data('abt_buy', model_name, f1)

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'amazon_google.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'amazon_google.json'),
                                  predictor=predictor,
                                  arguments=arguments)
    results_table.add_data('amazon_google', model_name, f1)

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'wdc_computers_large.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'wdc_computers_large.json'),
                                  predictor=predictor,
                                  standardizer_init=WDCDatasetStandardizer,
                                  arguments=arguments)
    results_table.add_data('wdc_computers_large', model_name, f1)

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'proprietary.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'proprietary.json'),
                                  predictor=predictor,
                                  standardizer_init=JSONLStandardizer,
                                  arguments=arguments)
    results_table.add_data('proprietary', model_name, f1)

    model_name, f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'proprietary_scarce.json'),
                                  preproc_config=os.path.join('configs',
                                                              'model_specific',
                                                              preproc_for_model,
                                                              'proprietary_scarce.json'),
                                  predictor=predictor,
                                  standardizer_init=CSVNoSplitStandardizer,
                                  arguments=arguments)
    results_table.add_data('proprietary', model_name, f1)


def run_baseline_experiments(arguments: ExperimentsArgumentParser):
    if not arguments.debug:
        wandb.init(project="master-thesis", entity="damianr13")
    f1_table = wandb.Table(columns=['experiment', 'model', 'score'])

    run_experiments_for_predictor(
        predictor=WordCoocPredictor(os.path.join('configs', 'model_train', 'word_cooc.json')),
        results_table=f1_table,
        arguments=arguments)
    #
    # run_experiments_for_predictor(predictor=AllMatchPredictor(), preproc_for_model='word_cooc', results_table=f1_table)
    #
    # run_experiments_for_predictor(predictor=NoMatchPredictor(), preproc_for_model='word_cooc', results_table=f1_table)
    #
    # run_experiments_for_predictor(predictor=BalancedPredictor(), preproc_for_model='word_cooc', results_table=f1_table)
    #
    # run_experiments_for_predictor(predictor=ClassDistributionAwarePredictor(),
    #                               preproc_for_model='word_cooc',
    #                               results_table=f1_table)

    if not arguments.debug:
        wandb.log({"f1_scores": f1_table})
    else:
        print(f1_table.data)


class ExperimentConfig(BaseModel):
    stand_path: str
    proc_path: str
    predictor_path: str
    standardizer: str
    known_clusters: bool = False


def standardizer_for_name(name: str, config_path: str) -> BaseStandardizer:
    if name == 'jsonl':
        return JSONLStandardizer(config_path=config_path)

    if name == 'wdc':
        return WDCDatasetStandardizer(config_path=config_path)

    if name == 'relational':
        return RelationalDatasetStandardizer(config_path=config_path)

    if name == 'csv_no_split':
        return CSVNoSplitStandardizer(config_path=config_path)

    raise Exception("Unknown standardizer requested")


def run_single_ditto_experiment(experiment_config: ExperimentConfig,
                                arguments: ExperimentsArgumentParser):
    seed_all(42)
    standardizer = standardizer_for_name(experiment_config.standardizer, experiment_config.stand_path)
    standardizer.preprocess()

    preprocessor = DittoPreprocessor(config_path=experiment_config.proc_path)

    target_split = standardizer.config.target_location.split('/')
    directory = target_split[-1] if target_split[-1] != '' else target_split[-2]
    default_preproc_target = os.path.join('data', 'processed', 'ditto', directory)

    preprocessor.preprocess(original_location=standardizer.config.target_location,
                            target_location=default_preproc_target)

    train_set = pd.read_csv(os.path.join(default_preproc_target, 'train.csv'))
    valid_set = pd.read_csv(os.path.join(default_preproc_target, 'valid.csv'))
    test_set = pd.read_csv(os.path.join(default_preproc_target, 'test.csv'))

    predictor = DittoPredictor(config_path=experiment_config.predictor_path, report=not arguments.debug, seed=42)
    predictor.train(train_set, valid_set, arguments=arguments)

    if not arguments.no_train:
        f1 = predictor.test(test_set)
        print(f'Finished with resulting f1 {f1}')

    if wandb.run:
        wandb.run.finish()


def run_single_supcon_experiment(experiment_config: ExperimentConfig,
                                 arguments: ExperimentsArgumentParser):
    seed_all(42)

    known_clusters = experiment_config.known_clusters
    standardizer = standardizer_for_name(experiment_config.standardizer, experiment_config.stand_path)
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

    if not arguments.no_train:
        f1 = predictor.test(test_set)
        print(f'Finished with resulting f1 {f1}')

    if wandb.run:
        wandb.run.finish()


def run_experiments(arguments: ExperimentsArgumentParser,
                    experiments: List[Dict],
                    experiment_function: Callable[[ExperimentConfig, ExperimentsArgumentParser], None]):
    for exp in experiments:
        experiment_config = ExperimentConfig.parse_obj(exp)
        experiment_function(experiment_config, arguments)


def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(os.path.join(path, 'train.csv')),\
            pd.read_csv(os.path.join(path, 'valid.csv')),\
            pd.read_csv(os.path.join(path, 'test.csv'))


def launch_secondary_sequence(arguments: ExperimentsArgumentParser):
    experiments = [
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'proprietary_scarce.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'proprietary_scarce.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive',
                                           'frozen_no-aug_batch-pt128_proprietary-scarce.json'),
            "standardizer": "csv_no_split",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'proprietary_scarce.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'proprietary_scarce.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive',
                                           'unfreeze_no-aug_batch-pt128_proprietary-scarce.json'),
            "standardizer": "csv_no_split",
            "known_clusters": False
        },
    ]

    run_experiments(arguments=arguments, experiments=experiments, experiment_function=run_single_supcon_experiment)


if __name__ == "__main__":
    start = datetime.now()
    args = ExperimentsArgumentParser().parse_args()

    print(os.getcwd())
    seed_all(42)
    torch.cuda.seed_all()

    if args.secondary_sequence:
        launch_secondary_sequence(args)
        exit(0)

    ditto_experiments = [
        # =========================== wdc_computers_medium ===========================================
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample75_wdc-computers-medium.json'),
            "standardizer": "wdc"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample50_wdc-computers-medium.json'),
            "standardizer": "wdc"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample25_wdc-computers-medium.json'),
            "standardizer": "wdc"
        },
        # ============================================ amazon_google =================================
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample75_amazon-google.json'),
            "standardizer": "relational"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_amazon-google.json'),
            "standardizer": "relational"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample50_amazon-google.json'),
            "standardizer": "relational"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample25_amazon-google.json'),
            "standardizer": "relational"
        },
        # # ====================================== abt_buy ========================================
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample75_abt-buy.json'),
            "standardizer": "relational"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_abt-buy.json'),
            "standardizer": "relational"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample50_abt-buy.json'),
            "standardizer": "relational"
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'ditto', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'ditto',
                                           'ditto_sample25_abt-buy.json'),
            "standardizer": "relational"
        }
    ]

    run_experiments(args, ditto_experiments, run_single_ditto_experiment)

    supcon_experiments = [
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample25_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample25_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive',
                                           'unfrozen_no-aug_batch-pt128_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample75_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample75_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample75_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample50_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample50_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'wdc_computers_medium_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'wdc_computers_medium.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample50_wdc-computers-medium.json'),
            "standardizer": "wdc",
            "known_clusters": True
        },
        # # ===================================== amazon google ====================================
        # # -----------------------------------50--------------------------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample50_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample50_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample50_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        # ------------------------------------------------25----------------------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample25_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample25_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample25_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive',
                                           'unfrozen_no-aug_batch-pt128_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        # ------------------------------------------------75----------------------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample75_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample75_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'amazon_google_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'amazon_google.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample75_amazon-google.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        # ============================================abt_buy========================================
        # --------------------------------------------50-------------------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample50_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample50_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.50.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample50_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        # -------------------------------------------------25------------------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample25_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample25_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.25.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample25_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        # -------------------------------------------------75------------------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'frozen_no-aug_batch-pt128_sample75_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfreeze_no-aug_batch-pt128_sample75_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy_0.75.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive', 'sampled',
                                           'unfrozen_no-aug_batch-pt128_sample75_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
        # ---------------------------------------full ---------------------------------
        {
            "stand_path": os.path.join('configs', 'stands_tasks', 'abt_buy.json'),
            "proc_path": os.path.join('configs', 'model_specific', 'contrastive', 'abt_buy.json'),
            "predictor_path": os.path.join('configs', 'model_train', 'contrastive',
                                           'unfrozen_no-aug_batch-pt128_abt-buy.json'),
            "standardizer": "relational",
            "known_clusters": False
        },
    ]
    run_experiments(args, supcon_experiments, run_single_supcon_experiment)

    # run_baseline_experiments()

    end = datetime.now()

    print(f"Execution took {end - start} ms")

    if args.debug:
        PerformanceWatcher.get_instance().print_stats()

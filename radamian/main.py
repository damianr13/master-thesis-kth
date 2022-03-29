import os
from typing import Callable

import pandas as pd
import wandb

from radamian.predictors.base import WordCoocPredictor, BasePredictor
from radamian.preprocess.definitions import BasePreprocessor
from radamian.preprocess.model_specific import WordCoocPreprocessor
from radamian.preprocess.standardize import RelationalDatasetStandardizer, WDCDatasetStandardizer


def run_pipeline(stand_config: str, preproc_config: str, predictor_config: str,
                 standardizer_init: Callable[[str], BasePreprocessor] = RelationalDatasetStandardizer,
                 preprocessor_init: Callable[[str], BasePreprocessor] = WordCoocPreprocessor,
                 predictor_init: Callable[[str], BasePredictor] = WordCoocPredictor) -> float:
    standardizer = standardizer_init(stand_config)
    preprocessor = preprocessor_init(preproc_config)
    predictor = predictor_init(predictor_config)

    standardizer.preprocess()
    preprocessor.preprocess()
    train_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(preprocessor.config.target_location, 'test.csv'))

    predictor.train(train_df, valid_df)
    return predictor.test(test_df)


def main():
    wandb.init(project="master-thesis", entity="damianr13")
    f1_table = wandb.Table(columns=['experiment', 'model', 'score'])

    f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'abt_buy.json'),
                      preproc_config=os.path.join('configs', 'model_specific', 'word_cooc', 'abt_buy.json'),
                      predictor_config=os.path.join('configs', 'model_train', 'word_cooc.json'))
    f1_table.add_data('abt_buy', 'word_cooc', f1)

    f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'amazon_google.json'),
                      preproc_config=os.path.join('configs', 'model_specific', 'word_cooc', 'amazon_google.json'),
                      predictor_config=os.path.join('configs', 'model_train', 'word_cooc.json'))
    f1_table.add_data('amazon_google', 'word_cooc', f1)

    f1 = run_pipeline(stand_config=os.path.join('configs', 'stands_tasks', 'wdc_computers_large.json'),
                      preproc_config=os.path.join('configs', 'model_specific', 'word_cooc', 'wdc_computers_large.json'),
                      predictor_config=os.path.join('configs', 'model_train', 'word_cooc.json'),
                      standardizer_init=WDCDatasetStandardizer)
    f1_table.add_data('wdc_computers_large', 'word_cooc', f1)

    wandb.log({"f1_scores": f1_table})


def testing_stuff():
    WDCDatasetStandardizer(os.path.join('configs', 'stands_tasks', 'wdc_computers_large.json')).preprocess()
    print('Finished')


if __name__ == "__main__":
    print(os.getcwd())
    main()

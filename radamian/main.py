import os

import pandas as pd
import wandb
from sklearn.metrics import f1_score

from radamian.predictors.base import WordCoocPredictor
from radamian.preprocess.model_specific import WordCoocPreprocessor
from radamian.preprocess.standardize import RelationalDatasetStandardizer


def main():
    wandb.init(project="master-thesis", entity="damianr13")

    RelationalDatasetStandardizer(os.path.join('configs', 'stands_tasks', 'abt_buy.json')).preprocess()
    RelationalDatasetStandardizer(os.path.join('configs', 'stands_tasks', 'amazon_google.json')).preprocess()

    WordCoocPreprocessor(os.path.join('configs', 'model_specific', 'word_cooc', 'abt_buy.json')).preprocess()

    train_set = pd.read_csv('data/preprocessed/word_cooc/abt_buy/train.csv')
    test_set = pd.read_csv('data/preprocessed/word_cooc/abt_buy/valid.csv')

    predictor = WordCoocPredictor(os.path.join('configs', 'model_train', 'word_cooc', 'abt_buy.json'))
    predictor.train(train_set, test_set)

    test_set = pd.read_csv('data/preprocessed/word_cooc/abt_buy/test.csv')
    predictor.test(test_set)


if __name__ == "__main__":
    print(os.getcwd())
    main()

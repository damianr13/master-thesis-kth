import numpy as np
import pandas as pd

from pandas import DataFrame
from src.predictors.base import BasePredictor
from sklearn.metrics import f1_score


class AllMatchPredictor(BasePredictor):

    def __init__(self):
        super(AllMatchPredictor, self).__init__("all_positives")

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        # no need to train
        pass

    def test(self, test_set) -> float:
        return f1_score(test_set['label'], np.ones(len(test_set)))


class NoMatchPredictor(BasePredictor):

    def __init__(self):
        super(NoMatchPredictor, self).__init__("all_negatives")

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        pass

    def test(self, test_set) -> float:
        return f1_score(test_set['label'], np.zeros(len(test_set)))


class BalancedPredictor(BasePredictor):

    def __init__(self):
        super(BalancedPredictor, self).__init__("balanced_predictor")

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        pass

    def test(self, test_set) -> float:
        np.random.seed(13)
        prediction = np.rint(np.random.rand(len(test_set)))
        return f1_score(test_set['label'], prediction)


class ClassDistributionAwarePredictor(BasePredictor):
    positives_proportion: float

    def __init__(self):
        super(ClassDistributionAwarePredictor, self).__init__("class_distribution_aware_predictor")

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        full_set = pd.concat((train_set, valid_set))
        true_matches_count = len(full_set[full_set['label'] == 1])
        self.positives_proportion = true_matches_count / len(full_set)

    def test(self, test_set) -> float:
        np.random.seed(13)
        positives_count = int(self.positives_proportion * len(test_set))
        positives = np.ones(positives_count)
        negatives = np.zeros(len(test_set) - positives_count)

        prediction = np.concatenate((positives, negatives))
        np.random.shuffle(prediction)

        return f1_score(test_set['label'], prediction)

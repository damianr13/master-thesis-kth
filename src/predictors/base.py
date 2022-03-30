import functools
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src import utils


class WordCoocClassifierConfig(BaseModel):
    name: str
    params: Dict[str, List]

    def generate_model(self, parameters=None):
        if parameters is None:
            parameters = {}

        if self.name == "bernoulli":
            return BernoulliNB(**parameters)
        elif self.name == "xgboost":
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric='error', **parameters)
        elif self.name == "random_forest":
            return RandomForestClassifier(**parameters)
        elif self.name == "decision_tree":
            return DecisionTreeClassifier(**parameters)
        elif self.name == "linear_svc":
            return LinearSVC(**parameters)
        elif self.name == "logistic_regression":
            return LogisticRegression(**parameters)
        else:
            raise Exception(f"Unknown model name {self.name}!")


class WordCoocConfig(BaseModel):
    classifiers: List[WordCoocClassifierConfig]


class BasePredictor(ABC):
    @abstractmethod
    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        pass

    @abstractmethod
    def test(self, test_set) -> float:
        pass


class WordCoocPredictor(BasePredictor):
    def __init__(self, config_path: str):
        self.config: WordCoocConfig = utils.load_as_object(config_path, WordCoocConfig.parse_obj)
        self.classifier = self.config.classifiers[0].generate_model()
        self.count_model = CountVectorizer(ngram_range=(1, 1), binary=True, min_df=2)
        self.train_features = []
        self.train_labels = []

        self.words = []

    def __extract_word_cooc(self, df: DataFrame) -> ndarray:
        left_counts = self.count_model.transform(df['ltext'])
        right_counts = self.count_model.transform(df['rtext'])

        word_cooc = left_counts.multiply(right_counts)
        return word_cooc

    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        full_corpus = pd.Series(train_set[['ltext', 'rtext']].fillna('').values.tolist()).str.join(' ').tolist()
        self.count_model.fit(full_corpus)

        full_set = pd.concat([train_set, valid_set])
        train_valid_split = PredefinedSplit(test_fold=np.concatenate((np.full(len(train_set), -1),
                                                                      np.full(len(valid_set), 0))))

        best_f1 = 0
        features = self.__extract_word_cooc(full_set)

        self.train_features = features
        self.train_labels = full_set['label'].tolist()
        for candidate_conf in self.config.classifiers:
            print(f'Working with model: {candidate_conf.name}\n')
            classifier = candidate_conf.generate_model()
            param_combinations = functools.reduce(
                lambda a, b: a * b,
                [len(p) for p in candidate_conf.params.values()],
                1
            )
            param_combinations = min(500, param_combinations)
            print(f'Testing {param_combinations} combination(s)')

            model_with_params = RandomizedSearchCV(cv=train_valid_split,
                                                   estimator=classifier,
                                                   param_distributions=candidate_conf.params,
                                                   random_state=13,
                                                   n_jobs=4,
                                                   scoring='f1',
                                                   n_iter=param_combinations,
                                                   pre_dispatch=8,
                                                   return_train_score=True)

            model_with_params.fit(self.train_features, self.train_labels)

            f1 = model_with_params.best_score_
            if f1 > best_f1:
                best_f1 = f1
                self.classifier = model_with_params

    def test(self, df: DataFrame) -> float:
        features = self.__extract_word_cooc(df)
        prediction = self.classifier.predict(features)
        f1 = f1_score(df['label'], prediction)
        return f1

    def predict(self, df: DataFrame) -> List:
        features = self.__extract_word_cooc(df)
        return self.classifier.predict(features)



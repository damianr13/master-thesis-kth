from abc import ABC, abstractmethod

from pandas import DataFrame


class BasePredictor(ABC):
    name: str

    def __init__(self, name):
        self. name = name

    @abstractmethod
    def train(self, train_set: DataFrame, valid_set: DataFrame) -> None:
        pass

    @abstractmethod
    def test(self, test_set) -> float:
        pass

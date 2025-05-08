from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Encoder(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("This encoder does not support inverse_transform")

class Decoder(ABC):
    @abstractmethod
    def decode(self, y: np.ndarray) -> np.ndarray:
        pass

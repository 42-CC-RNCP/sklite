from typing import Optional, Tuple
from pandas import DataFrame
from sklite.core.transformer import Transformer


class StandarScaler(Transformer):
    def __init__(self, columns: Optional[Tuple[str]] = None):
        self.columns = columns
        self.means = {}
        self.stds = {}

    def fit(self, X: DataFrame):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["number"]).columns.tolist()
        for col in self.columns:
            self.means[col] = X[col].mean()
            self.stds[col] = X[col].std()

    def transform(self, X: DataFrame) -> DataFrame:
        X_new = X.copy()
        for col in self.columns:
            X_new[col] = (X[col] - self.means[col]) / self.stds[col]
        return X_new
    
    def inverse_transform(self, X: DataFrame) -> DataFrame:
        X_new = X.copy()
        for col in self.columns:
            X_new[col] = (X[col] * self.stds[col]) + self.means[col]
        return X_new
    

class MinMaxScaler(Transformer):
    def __init__(self, columns: Optional[Tuple[str]] = None):
        self.columns = columns
        self.mins = {}
        self.maxs = {}

    def fit(self, X: DataFrame):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["number"]).columns.tolist()
        for col in self.columns:
            self.mins[col] = X[col].min()
            self.maxs[col] = X[col].max()

    def transform(self, X: DataFrame) -> DataFrame:
        X_new = X.copy()
        for col in self.columns:
            X_new[col] = (X[col] - self.mins[col]) / (self.maxs[col] - self.mins[col])
        return X_new
    
    def inverse_transform(self, X: DataFrame) -> DataFrame:
        X_new = X.copy()
        for col in self.columns:
            X_new[col] = (X[col] * (self.maxs[col] - self.mins[col])) + self.mins[col]
        return X_new

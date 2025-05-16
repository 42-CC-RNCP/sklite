from typing import Optional, Tuple
from pandas import DataFrame
from sklite.core.transformer import Transformer


class LabelEncoder(Transformer):
    def __init__(self, columns: Optional[Tuple[str]] = None):
        self.columns = columns
        self.label_maps = {}

    def fit(self, X: DataFrame):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object"]).columns.tolist()
        for col in self.columns:
            unique_values = sorted(X[col].unique())
            self.label_maps[col] = {val: idx for idx, val in enumerate(unique_values)}
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X_new = X.copy()
        for col in self.columns:
            X_new[col] = X[col].map(self.label_maps[col])
        return X_new

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        X_new = X.copy()
        for col in self.columns:
            inv_mapping = {v: k for k, v in self.label_maps[col].items()}
            X_new[col] = X[col].map(inv_mapping)
        return X_new
    
    def __repr__(self) -> str:
        # print the mapping of each column
        return f"LabelEncoder(label_maps={self.label_maps})"

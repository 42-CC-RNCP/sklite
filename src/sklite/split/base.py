import numpy as np
from typing import Generator
from abc import ABC, abstractmethod


class Splitter(ABC):        
    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray = None):
        pass

import numpy as np
from abc import *
from typing import Callable
from dolfinx.fem.function import Constant


class TopOptProblem(ABC):
    @abstractmethod
    def define_mesh(self):
        pass
    @abstractmethod
    def define_BC(self) -> (Callable, Constant):
        pass
    @abstractmethod
    def define_LC(self) -> (list[tuple[int,Callable]], np.ndarray):
        pass
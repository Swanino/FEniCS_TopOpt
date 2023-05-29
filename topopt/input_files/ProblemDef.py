from abc import *

class TopOptProblem(ABC):
    @abstractmethod
    def define_mesh(self):
        pass
    @abstractmethod
    def define_BC(self):
        pass
    @abstractmethod
    def define_LC(self):
        pass
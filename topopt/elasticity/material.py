import numpy as np
from abc import *

class Material(ABC):
    @abstractmethod
    def get_modulus_tensor(self) -> np.ndarray:
        pass
    @abstractmethod
    def __str__(self) -> str:
        pass
    
class Material_el_lin_iso(Material):
    def __init__(self, dim:int=2, lmd:float=0.6, mu:float=0.4) -> None:
        self.type = "isotropic linear elastic"
        self.lmd, self.mu = lmd, mu
        self.dim = dim
        self.lame_to_Ev()
    def lame_to_Ev(self):
        if self.dim == 2:
            self.E = self.mu * (3 * self.lmd + 2 * self.mu) / (self.lmd + self.mu)
            self.nu = self.lmd / (2 * (self.lmd + self.mu))
        elif self.dim == 3:
            self.E = 4.0*self.mu * (self.lmd + self.mu) / (self.lmd + 2.*self.mu)
            self.nu = self.lmd / (self.lmd + 2.*self.mu)
    def get_modulus_tensor(self) -> np.ndarray:
        if self.dim == 2:
            print("2D plane stress")
            Cijkl = self.E / (1.0 - self.nu ** 2) * np.array([[1.0, self.nu, 0.0], [self.nu, 1.0, 0.0], [0.0, 0.0, (1.0 - self.nu) / 2.0]])
        elif self.dim == 3:
            Cijkl = self.E / (1.0 + self.nu) / (1.0 - 2.0 * self.nu) * np.array([[1.0 - self.nu, self.nu, self.nu, 0.0, 0.0, 0.0],
                                                                               [self.nu, 1.0 - self.nu, self.nu, 0.0, 0.0, 0.0],
                                                                               [self.nu, self.nu, 1.0 - self.nu, 0.0, 0.0, 0.0],
                                                                               [0.0, 0.0, 0.0, 0.5 - self.nu, 0.0, 0.0],
                                                                               [0.0, 0.0, 0.0, 0.0, 0.5 - self.nu, 0.0],
                                                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 - self.nu]])
        return Cijkl
    def __str__(self) -> str:
        return f"isotropic, linear elastic material of dim={self.dim}\n(E={self.E}, nu={self.nu}, lmd={self.lmd}, mu={self.mu})"
    
def __test__():
    mat = Material_el_lin_iso(dim=2, lmd=0.6, mu=0.4)
    print(mat)
    print(mat.get_modulus_tensor())

if __name__ == "__main__":
    __test__()
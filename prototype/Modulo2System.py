import numpy as np
from typing import List
from bitarray import bitarray

class SparseModulo2System:
    def __init__(self,
                 solution_size: int):
        # solution_size: max_hash_range + max_code_length
        # equations: {id: (nonzero_index_0, nonzero_index_1, ...)}
        self._equations = {}
        self._constants = {}
        self._solution_size = solution_size
        # The "degree" of a variable is a graph-theoretic term referring to the
        # number of equations in which the variable participates.
        self._degree = np.zeros(self._solution_size)

    def addEquation(self,
                    equation_id: int,
                    participating_variables: List[int],
                    constant: int):
        if constant < 0 or constant > 1:
            raise ValueError(f"Constant must be 0 or 1.")
        if equation_id in self._equations:
            raise ValueError(f"Equation id {equation_id:d} is already present "
                              "in the system.")
        for var in participating_variables:
            if var >= self._solution_size:
                raise ValueError(f"Invalid variable id: id {var:d} >= "
                                 f"{self._solution_size:d}.")
        self._equations[equation_id] = participating_variables
        self._constants[equation_id] = constant
        for var in participating_variables:
            self._degree[var] += 1

    def getEquation(self,
                    equation_id: int):
        return (self._equations[equation_id], self._constants[equation_id])

    @property
    def shape(self):
        # Returns (num_equations, num_variables). Read-only.
        return (len(self._equations), self._solution_size)

    @property
    def equation_ids(self):
        # Returns a sorted list of equation ids. Read-only.
        return list(self._equations.keys())


class DenseModulo2System:
    def __init__(self,
                 solution_size: int):
        # Note: This can be represented more efficiently via a 2D bit array,
        # but we are using a dictionary for readability. The difficulty is that
        # in a larger system, the index of a row in the DenseModulo2System may
        # not be the same as its equation_id (e.g. after peeling).
        self._equations = {}
        self._constants = {}
        self._solution_size = solution_size
    
    def addEquation(self,
                    equation_id: int,
                    participating_variables: List[int],
                    constant: int):
        if constant < 0 or constant > 1:
            raise ValueError(f"Constant must be 0 or 1.")
        if equation_id in self._equations:
            raise ValueError(f"Equation id {equation_id:d} is already present "
                              "in the system.")
        for var in participating_variables:
            if var >= self._solution_size:
                raise ValueError(f"Invalid variable id: id {var:d} >= "
                                 f"{self._solution_size:d}.")

        equation = bitarray(self._solution_size)
        equation.setall(0)
        # Set the correct bits in the backing array.
        for var in participating_variables:
            equation[var] = 1

        self._equations[equation_id] = equation
        self._constants[equation_id] = constant

    def getEquation(self,
                    equation_id: int):
        return (self._equations[equation_id], self._constants[equation_id])

    @property
    def shape(self):
        # Returns (num_equations, num_variables). Read-only.
        return (len(self._equations), self._solution_size)

    @property
    def equation_ids(self):
        # Returns a sorted list of equation ids. Read-only.
        return list(self._equations.keys())

    def xorEquations(self,
                     equation_to_modify: int,
                     equation_to_xor: int):
        # Computes the XOR of the equation and constant associated with
        # equation_to_modify and equation_to_xor, and places the result into 
        # equation_to_modify.
        c_to_modify = self._constants[equation_to_modify]
        c_to_xor = self._constants[equation_to_xor]
        new_c = np.bitwise_xor(c_to_modify, c_to_xor)

        self._equations[equation_to_modify] ^= self._equations[equation_to_xor]
        self._constants[equation_to_modify] = new_c

    def swapEquations(self,
                     equation_id_1: int,
                     equation_id_2: int):
        temp_equation = self._equations[equation_id_1]
        self._equations[equation_id_1] = self._equations[equation_id_2]
        self._equations[equation_id_2] = temp_equation

        temp_constant = self._constants[equation_id_1]
        self._constants[equation_id_1] = self._constants[equation_id_2]
        self._constants[equation_id_2] = temp_constant

    def getFirstVar(self, equation_id: int) -> int:
        # returns the first non-zero bit index in equation_id's equation
        if not self._equations[equation_id].any():
            if self._constants[equation_id]:
                # In this case, we have a linearly dependent row
                raise UnsolvableSystemException
            else:
                # In this case, we have an identity row
                return self._solution_size
    
        return self._equations[equation_id].find(1)

    def isUnsolvable(self, equation_id: int) -> bool:
        # returns if the equation is all zeros and the constant is not 0
        isEmpty = not self._equations[equation_id].any()
        return isEmpty and self._constants[equation_id] != 0

    def isIdentity(self, equation_id: int) -> bool:
        # returns if the equation is all zeros and the constant IS 0
        isEmpty = not self._equations[equation_id].any()
        return isEmpty and self._constants[equation_id] == 0

    def equationToStr(self, equation_id: int) -> str:
        return f"{self._equations[equation_id].to01()} | {self._constants[equation_id]:d} (Equation [{equation_id:d}]"

    def systemToStr(self, equation_ids=None):
        if equation_ids is None:
            equation_ids = list(self._equations.keys())
            equation_ids.sort()
        system_str = ""
        for equation_id in equation_ids:
            system_str += self.equationToStr(equation_id) + "\n"
        return system_str


class UnsolvableSystemException(Exception):
    pass

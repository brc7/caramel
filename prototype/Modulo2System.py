from pandas import array
import spookyhash
import numpy as np

from typing import List
import gmpy2

# For usage example
import random
import string
import math

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
        # Note: Backing type must be little-endian.
        self._backing_type = np.dtype('<i4')

        # Determine the size of the backing array from the solution size
        # Each variable is 1 bit, so 8 * num_bytes variables / item in array
        self._num_variables_per_chunk = self._backing_type.itemsize * 8
        self._bitvector_size = int(math.ceil(
            self._solution_size / self._num_variables_per_chunk))
    
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
        backing_array = np.zeros(shape=self._bitvector_size,
                                 dtype=self._backing_type)
        # Set the correct bits in the backing array.
        for var in participating_variables:
            backing_array = self._update_bitvector(backing_array, var)
        self._equations[equation_id] = backing_array
        self._constants[equation_id] = constant

    def getEquation(self,
                    equation_id: int):
        return (self._equations[equation_id], self._constants[equation_id])

    @property
    def shape(self):
        # Returns (num_equations, num_variables). Read-only.
        return (len(self._equations), self._solution_size)

    @property
    def dtype(self):
        # Returns the dtype of the backing array. Read-only.
        return self._backing_type

    @property
    def bitvector_size(self):
        # Returns the size of the numpy bitvector 
        return self._bitvector_size
        
    def xorEquations(self,
                     equation_to_modify: int,
                     equation_to_xor: int):
        # Computes the XOR of the equation and constant associated with
        # equation_to_modify and equation_to_xor, and places the result into 
        # equation_to_modify.
        c_to_modify = self._constants[equation_to_modify]
        c_to_xor = self._constants[equation_to_xor]
        new_c = np.bitwise_xor(c_to_modify, c_to_xor)

        new_equation = np.bitwise_xor(self._equations[equation_to_modify],
                                      self._equations[equation_to_xor])

        self._equations[equation_to_modify] = new_equation
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

        # TODO fails if equation is all zeros. should we return np.iinfo(self._backing_type)?
        # how should we handle max values in the gaussian elimination?
    
        # TODO: np.where searches the whole array, and doesnt stop at the first 
        # sight of nonzero, can we optimize this?
        equation = self._equations[equation_id]
        first_nonzero_chunk_id = np.where(equation != 0)[0][0]
        chunk = equation[first_nonzero_chunk_id]

        # Using gmpy to find the index of the least significant bit. Sources:
        # https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
        # https://stackoverflow.com/questions/12078277/is-this-a-bug-in-gmpy2-or-am-i-mad
        index_of_first_set_bit_in_chunk = gmpy2.bit_scan1(gmpy2.mpz(chunk))
        return first_nonzero_chunk_id * self._num_variables_per_chunk + index_of_first_set_bit_in_chunk

    def isUnsolvable(self, equation_id: int) -> bool:
        # returns if the equation is all zeros and the constant is not 0
        isEmpty = not self._equations[equation_id].any()
        return isEmpty and self._constants[equation_id] != 0

    def isIdentity(self, equation_id: int) -> bool:
        # returns if the equation is all zeros and the constant IS 0
        isEmpty = not self._equations[equation_id].any()
        return isEmpty and self._constants[equation_id] == 0

    def equationToStr(self, equation_id: int) -> str:
        return self.bitArrayToStr(self._equations[equation_id])
    
    def bitArrayToStr(self, bitarray):
        array_str = ""
        for chunk in bitarray:
            # [2:] to remove the "0b" at the beginning of the bit string
            # [::-1] to reverse it since we work with little-endian ordering
            chunk_str = bin(chunk)[2:][::-1]
            # we either pad to round out the current chunk or the whole solution
            num_padding_zeroes = min(self._solution_size - len(array_str), self._num_variables_per_chunk) - len(chunk_str)
            chunk_str += "0" * num_padding_zeroes
            array_str += chunk_str
        return array_str

    def _update_bitvector(self, array, bit_id, value=1):
        chunk_id = bit_id // self._num_variables_per_chunk
        if chunk_id >= len(array):
            raise ValueError(f"Tried to set chunk id {chunk_id:d} in backing "
                             f"array of size {len(array):d}.")
        # TODO: Replace and test with more efficient, and equivalent
        # num_left_shifts = bit_id - chunk_id * self._num_variables_per_chunk
        num_left_shifts = bit_id % self._num_variables_per_chunk
        chunk = np.array([1], dtype=self._backing_type)
        chunk = np.left_shift(chunk, num_left_shifts)
        if value:
            array[chunk_id] = np.bitwise_or(array[chunk_id], chunk)
        else:
            chunk = np.bitwise_not(chunk)
            array[chunk_id] = np.bitwise_and(array[chunk_id], chunk)
        return array


class UnsolvableSystemException(Exception):
    pass


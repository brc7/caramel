import spookyhash
import numpy as np

from typing import List

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
        self._equations{equation_id} = participating_variables
        self._constants{equation_id} = constant

    def getEquation(self,
                    equation_id: int):
        return (self._equations{equation_id}, self._constants{equation_id})


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
                    participating_variables: List[int]):
        if constant < 0 or constant > 1:
            raise ValueError(f"Constant must be 0 or 1.")
        if equation_id in self._equations:
            raise ValueError(f"Equation id {equation_id:d} is already present "
                              "in the system.")
        for var in participating_variables:
            if var >= self._solution_size:
                raise ValueError(f"Invalid variable id: id {var:d} >= "
                                 f"{self._solution_size:d}.")
        backing_array = np.zeros(size=self._bitvector_size,
                                 dtype=self._backing_type)
        # Set the correct bits in the backing array.
        for var in participating_variables:
            backing_array = self._update_bitvector(backing_array, var)
        self._equations{equation_id} = backing_array
        self._constants{equation_id} = constant

    def getEquation(self,
                    equation_id: int):
        return (self._equations{equation_id}, self._constants{equation_id})

    def _update_bitvector(self, backing_array, bit_index, value=1):
        chunk_id = bit_id // self._num_variables_per_chunk
        if chunk_id >= len(backing_array):
            raise ValueError(f"Tried to set chunk id {chunk_id:d} in backing "
                             f"array of size {len(backing_array):d}.")
        # TODO: Replace and test with more efficient, and equivalent
        # num_left_shifts = bit_id - chunk_id * self._num_variables_per_chunk
        num_left_shifts = bit_id % self._num_variables_per_chunk
        chunk = np.array([1], dtype=self._backing_type)
        chunk = np.left_shift(chunk, num_left_shifts)
        if value:
            backing_array[chunk_id] = np.bitwise_or(array[chunk_id], chunk)
        else:
            chunk = np.bitwise_not(chunk)
            backing_array[chunk_id] = np.bitwise_and(array[chunk_id], chunk)
        return array

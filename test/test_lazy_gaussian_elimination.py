import random

import pytest
from caramel.LazyGaussianElimination import lazy_gaussian_elimination
from caramel.Modulo2System import (SparseModulo2System,
                                   UnsolvableSystemException)


@pytest.mark.skip(reason="Need to clean up exception handling in the test")
def test_random_system(num_equations, num_variables, verbose=0):
    sparse_system = SparseModulo2System(num_variables)
    equation_ids = list(range(num_equations))
    if verbose >= 1:
        print('Constructing linear system...')
    for equation_id in equation_ids:
        variables = [random.randrange(0, num_variables) for _ in range(3)]
        constant = random.choice([0,1])
        sparse_system.addEquation(equation_id, variables, constant)
    if verbose >= 1:
        print('Solving linear system...')
    try:
        state = lazy_gaussian_elimination(sparse_system,
                                        equation_ids,
                                        verbose=verbose)
    except UnsolvableSystemException as e:
        return True
    return True


def test_unsolvable_pair(verbose=0):
    # Tests the case where two equations have incompatible constants.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [5,8,9], 1)  # Unsolvable duplicate equation.
    equation_ids = [0,1,2,3,4,5]
    try: 
        _ = lazy_gaussian_elimination(sparse_system,
                                      equation_ids,
                                      verbose=verbose)
    except UnsolvableSystemException as e:
        return True
    return False

def test_solvable_system(verbose=0):
    # Tests a system that is known to be solvable.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    equation_ids = [0,1,2,3,4,5]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True


def test_system_subset(verbose=0):
    # Tests a system that is known to be solvable.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    equation_ids = [0,2,3,5]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True

def test_unsolvable_pair(verbose=0):
    # Tests the case where two equations have incompatible constants.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [5,8,9], 1)  # Unsolvable duplicate equation.
    equation_ids = [0,1,2,3,4,5]
    try: 
        _ = lazy_gaussian_elimination(sparse_system,
                                      equation_ids,
                                      verbose=verbose)
    except UnsolvableSystemException as e:
        return True
    return False


def test_active_system(verbose=0):
    # Tests a system that is known to be solvable, but has an active core.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    sparse_system.addEquation(6, [2,8,9], 1)
    sparse_system.addEquation(7, [0,7,9], 1)
    sparse_system.addEquation(8, [1,7,9], 0)
    sparse_system.addEquation(9, [1,2,9], 0)
    equation_ids = [0,1,2,3,4,5,6,7,8,9]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True


def test_active_with_duplicate_system(verbose=0):
    # Tests a system that is known to be solvable, but has an active core and
    # includes duplicate vertices in the hypergraph specification.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    sparse_system.addEquation(6, [2,8,9], 1)
    sparse_system.addEquation(7, [0,7,9], 1)
    sparse_system.addEquation(8, [1,7,9], 0)
    sparse_system.addEquation(9, [1,2,9], 0)
    equation_ids = [0,1,2,3,4,5,6,7,8,9]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True


def test_solvable_with_duplicates_system(verbose=0):
    # Tests a system that is known to be solvable, but includes duplicate
    # vertices in the hypergraph specification.
    num_variables = 9
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [0,1,2], 1)
    sparse_system.addEquation(1, [2,3,4], 1)
    sparse_system.addEquation(2, [3,4,5], 0)
    sparse_system.addEquation(3, [5,6,7], 1)
    sparse_system.addEquation(4, [0,8,8], 1)
    equation_ids = [0,1,2,3,4]

    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True


@pytest.mark.skip(reason="Need to clean up exception handling in the test")
def test_random_inputs():
    hypergraph_dimensions = [(random.randint(10,100),random.randint(10,100))
                                for _ in range(1000)]

    for n, m in hypergraph_dimensions:
        _ = test_random_system(n, m, verbose=0)

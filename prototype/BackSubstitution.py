import numpy as np
from Modulo2System import *
from LazyGaussianElimination import *
from HypergraphPeeler import *
from GaussianElimination import *

def solve_lazy_from_dense(dense_equation_ids,
                          solved_equation_ids,
                          solved_variable_ids,
                          dense_system,
                          dense_solution,
                          verbose=0):
    # Solve the lazy gaussian elimination variables using the solutions to the
    # dense linear system (that were found using regular gaussian elimination).
    # 
    # dense_solution: The solution to dense_system, from GaussianElimiation.
    
    assert len(solved_equation_ids) == len(solved_variable_ids)
    for equation_id, variable_id in zip(solved_equation_ids,solved_variable_ids):
        # 1. The only non-dense variable that still needs to be solved is 
        # variable_id (by the invariants of the lazy gaussian elimination)
        # The solution is zero at this index, so the bit to set is just 
        # the constant XOR <equation_coefficients, solution_so_far>
        equation, constant = dense_system.getEquation(equation_id)
        value = np.bitwise_xor(constant,
                               scalarProduct(equation, dense_solution)) % 2
        value = np.bitwise_and(1, value)
        if verbose >= 2:
            print(f"[Equation {equation_id}] solving for [Variable {variable_id}]"
                f": Value = {value}, constant = {constant}")
        dense_solution[variable_id] = value
    return dense_solution


def solve_peeled_from_dense(peeled_equation_ids,
                            var_solution_order,
                            dense_system,
                            dense_solution,
                            verbose=0):
    # Solve the peeled hypergraph representation of the linear system using the
    # solution to the unpeelable 2-core of the system (dense_solution).
    # 
    # dense_solution: The solution to the system, from LazyGaussianElimination
    # or from GaussianElimination.

    # 1. Peeled IDs contains a list of equation IDs in solution-order.
    # Var solution order contains a list of variables that need to be solved.

    for equation_id, variable_id in zip(peeled_equation_ids, var_solution_order):
        # Update dense_solution to include these.
        equation, constant = dense_system.getEquation(equation_id)
        # TODO: pPrformance tuning, this is highly inefficient
        value = np.bitwise_xor(constant,
                               scalarProduct(equation, dense_solution)) % 2
        value = np.bitwise_and(1, value)
        if verbose >= 2:
            print(f"[Equation {equation_id}] solving for [Variable {variable_id}]"
                f": Value = {value}, constant = {constant}")
        dense_solution[variable_id] = value
    return dense_solution


def sparse_to_dense(sparse_system, equation_ids=None):
    num_equations, num_variables = sparse_system.shape
    dense_system = DenseModulo2System(num_variables)
    if equation_ids is None:
        equation_ids = sparse_system.equation_ids
    for equation_id in equation_ids:
        vars_to_add, constant = sparse_system.getEquation(equation_id)
        dense_system.addEquation(equation_id, list(vars_to_add), constant)
    return dense_system


def test_lazy_from_dense_solvable_system(verbose=0):
    num_variables = 10
    equations = (([1,2,3],1),
                 ([3,4,5],1),
                 ([4,5,6],0),
                 ([6,7,8],1),
                 ([5,8,9],0),
                 ([0,8,9],1),
                 ([2,8,9],1),
                 ([0,7,9],1),
                 ([1,7,9],0),
                 ([1,2,9],0))
    equation_ids = list(range(len(equations)))
    sparse_system = SparseModulo2System(num_variables)

    for equation_id, (variables, constant) in enumerate(equations):
        sparse_system.addEquation(equation_id, variables, constant)
    try:
        state = lazy_gaussian_elimination(sparse_system,
                                        equation_ids,
                                        verbose=verbose)
        dense_ids, solved_ids, solved_vars, dense_system = state
        print("System (after lazy gaussian elimination): ")
        print(dense_system.systemToStr())
        dense_solution = gaussian_elimination(dense_system, dense_ids, verbose=verbose)
        print("System (after gaussian elimination): ")
        print(dense_system.systemToStr())
        print("Solution (after gaussian elimination): ")
        print(dense_solution.to01())
    except UnsolvableSystemException as e:
        return False
    dense_solution = solve_lazy_from_dense(dense_ids,
                                           solved_ids,
                                           solved_vars,
                                           dense_system,
                                           dense_solution)
    print("Solution (after lazy gaussian elimination): ")
    print(dense_solution.to01())

    # Check the solution
    original_system = sparse_to_dense(sparse_system)
    # Do the inner products explicitly for each row
    for equation_id in original_system.equation_ids:
        equation, constant = original_system.getEquation(equation_id)
        result = scalarProduct(equation, dense_solution) % 2 
        print(f"Equation {equation_id}: {result}, {constant}")


def test_peeled_from_dense_solvable_system(verbose=0):
    num_variables = 10
    equations = (([1,2,3],1),
                 ([3,4,5],1),
                 ([4,5,6],0),
                 ([6,7,8],1),
                 ([5,8,9],0),
                 ([0,8,9],1),
                 ([2,8,9],1),
                 ([0,7,9],1),
                 ([1,7,9],0),
                 ([1,2,9],0))
    equation_ids = list(range(len(equations)))
    sparse_system = SparseModulo2System(num_variables)

    for equation_id, (variables, constant) in enumerate(equations):
        sparse_system.addEquation(equation_id, variables, constant)
    try:
        state = peel_hypergraph(sparse_system, sparse_system.equation_ids)
        unpeeled_ids, peeled_ids, var_order, sparse_system = state

        dense_system = sparse_to_dense(sparse_system, unpeeled_ids)

        print("System (before gaussian elimination): ")
        print(dense_system.systemToStr())

        dense_solution = gaussian_elimination(dense_system, 
                                                unpeeled_ids, 
                                                verbose=verbose)
        print("System (after gaussian elimination): ")
        print(dense_system.systemToStr())
        print("Solution (after gaussian elimination): ")
        print(dense_solution.to01())
    except UnsolvableSystemException as e:
        print("SAD")
        return False

    dense_solution = solve_peeled_from_dense(
        peeled_ids, var_order, dense_system, dense_solution)

    print("Solution (after peeling back-substitution): ")
    print(dense_solution.to01())

    # Check the solution
    original_system = sparse_to_dense(sparse_system)
    # Do the inner products explicitly for each row
    for equation_id in original_system.equation_ids:
        equation, constant = original_system.getEquation(equation_id)
        result = scalarProduct(equation, dense_solution) % 2
        print(f"Equation {equation_id}: {result}, {constant}")

if __name__ == '__main__':
    test_lazy_from_dense_solvable_system(verbose=2)
    test_peeled_from_dense_solvable_system(verbose=2)

import numpy as np

from caramel.GaussianElimination import *
from caramel.HypergraphPeeler import *
from caramel.LazyGaussianElimination import *
from caramel.Modulo2System import *


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
        participating_vars, constant = sparse_system.getEquation(equation_id)
        # We should only add a variable to the equation in the dense system if
        # it appears an odd number of times. This is because we compute output
        # as XOR(solution[hash_1], solution[hash_2] ...). If hash_1 = hash_2 = 
        # variable_id, then XOR(solution[hash_1], solution[hash_2]) = 0 and
        # the variable_id did not actually participate in the solution.
        vars_to_add = set()
        for variable_id in participating_vars:
            if variable_id not in vars_to_add:
                vars_to_add.add(variable_id)
            else:
                vars_to_add.remove(variable_id)
        dense_system.addEquation(equation_id, list(vars_to_add), constant)
    return dense_system


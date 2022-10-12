from caramel.BackSubstitution import (scalarProduct, solve_lazy_from_dense,
                                      solve_peeled_from_dense, sparse_to_dense)
from caramel.GaussianElimination import gaussian_elimination
from caramel.HypergraphPeeler import peel_hypergraph
from caramel.LazyGaussianElimination import lazy_gaussian_elimination
from caramel.Modulo2System import (SparseModulo2System,
                                   UnsolvableSystemException)


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
        return False

    dense_solution = solve_peeled_from_dense(
        peeled_ids, var_order, sparse_system, dense_solution)

    print("Solution (after peeling back-substitution): ")
    print(dense_solution.to01())

    # Check the solution
    original_system = sparse_to_dense(sparse_system)
    # Do the inner products explicitly for each row
    for equation_id in original_system.equation_ids:
        equation, constant = original_system.getEquation(equation_id)
        result = scalarProduct(equation, dense_solution) % 2
        print(f"Equation {equation_id}: {result}, {constant}")

import random
import numpy as np
from caramel.BackSubstitution import sparse_to_dense, scalarProduct
from caramel.Modulo2System import SparseModulo2System, UnsolvableSystemException
from bin.construct import solve_modulo2_system


def create_random_matrix():
    num_equations = 10
    num_variables = 11

    sparse_system = SparseModulo2System(num_variables)
    numpy_matrix = np.zeros([num_equations, num_variables])
    numpy_constants = np.zeros(num_equations)

    for equation_id in range(num_equations):
        indices = [random.randint(0, num_variables - 1) for _ in range(3)]
        constant = random.randint(0, 1)
        sparse_system.addEquation(equation_id, indices, constant)

        for index in indices:
            numpy_matrix[equation_id][index] = 1
        numpy_constants[equation_id] = constant

    return sparse_system, numpy_matrix, numpy_constants


def verify_solution(original_sparse_system, solution):
    original_system = sparse_to_dense(original_sparse_system)
    # Do the inner products explicitly for each row
    for equation_id in original_system.equation_ids:
        equation, constant = original_system.getEquation(equation_id)
        result = scalarProduct(equation, solution) % 2
        if result != constant:
            return False
    return True


def test_modulo2_solver():
    solved_count = 0
    num_iters = 1000
    for _ in range(num_iters):
        try:
            sparse_system, numpy_matrix, numpy_constants = create_random_matrix()
            solution = solve_modulo2_system(sparse_system)
            solution_is_correct = verify_solution(sparse_system, solution)
            assert solution_is_correct
            solved_count += 1
        except UnsolvableSystemException as e:
            pass
    
    print(f"Produced correct solutions for {solved_count} / {num_iters} matrices.")


if __name__ == "__main__":
    test_modulo2_solver()

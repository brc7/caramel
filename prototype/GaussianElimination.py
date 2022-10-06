import numpy as np
import bitarray, bitarray.util
from Modulo2System import DenseModulo2System, UnsolvableSystemException

'''
We perform Gaussian Elimination by maintaining the state of each equation's 
"first_var". The first_var is the index of the first non-zero bit in the 
equation. Overall this algorithm works as follows:

    1. Calculate the first var for each equation in relevant_equation_ids.
    2. Iterate through all ordered pairs of equations and swap/xor them around 
        to get them into echelon form. We can break down each ordered pair of 
        equations into a Top equation and a Bot equation. The general steps in 
        this process are:

        A. Check if both equations have the same first var. If so then we set 
        Bot equation equal to Bot equation XORed with the Top equation.
        B. Verify that Top equation's first var is greater than Bot equation's
        first var. Otherwise, swap these two equations. 

    3. Back-substitution. Go backwards through the matrix (from bottom to top in
    the echelon form matrix) and set the bit of the solution to whatever 
    resolves the constant.

Throws UnsolvableSystemException.
'''

def gaussian_elimination(dense_system, relevant_equation_ids, verbose=True):
    first_vars = {}
    for equation_id in relevant_equation_ids:
        first_vars[equation_id] = dense_system.getFirstVar(equation_id)
    
    num_equations = len(relevant_equation_ids)
    for top_equation_index in range(num_equations - 1):
        for bot_equation_index in range(top_equation_index + 1, num_equations):
            top_equation_id = relevant_equation_ids[top_equation_index]
            bot_equation_id = relevant_equation_ids[bot_equation_index]

            if verbose:
                print(f"\nStarting Iteration:")
                print(f"  Top Equation: {dense_system.equationToStr(top_equation_id)}")
                print(f"  Bot Equation: {dense_system.equationToStr(bot_equation_id)}")

            if first_vars[top_equation_id] == first_vars[bot_equation_id]:
                # Leading 1 in top is above the leading 1 in bot, so
                # eliminate this variable from bot via xor operations.
                dense_system.xorEquations(bot_equation_id, top_equation_id)
                if verbose:
                    print(f"    Top and Bot have equal first vars. Set Bot = Top"
                          f" XOR Bot. Bot becomes: "
                          f"{dense_system.equationToStr(bot_equation_id)}")
                if dense_system.isUnsolvable(top_equation_id):
                    raise UnsolvableSystemException(f"Equation {equation_id:d}"
                                f"has all coefficients = 0 but constant is 1.")
                # If bot is an identity, first_vars[bot_equation_id] is 
                # num_variables (which skips the rest of the inner loop).
                first_vars[bot_equation_id] = dense_system.getFirstVar(bot_equation_id)

            if first_vars[top_equation_id] > first_vars[bot_equation_id]:
                if verbose:
                    print(f"    Swapping equations based on first vars.")
                    print(f"first vars: top: {first_vars[top_equation_id]}, bot: {first_vars[bot_equation_id]}")

                # TODO: this swap here is not nice because we swap everything
                # individually (equations, constants, first vars). lets
                # refactor this eventually to be nicer
                dense_system.swapEquations(top_equation_id, bot_equation_id)
                temp_first_vars = first_vars[top_equation_id]
                first_vars[top_equation_id] = first_vars[bot_equation_id]
                first_vars[bot_equation_id] = temp_first_vars

    if verbose:
        print("\nCompleted Echelon Form. Now doing back-substitution.")

    solution_size = dense_system.shape[1]
    solution = bitarray.bitarray(solution_size)
    solution.setall(0)
    for i in range(len(relevant_equation_ids) - 1, -1, -1):
        equation_id = relevant_equation_ids[i]
        equation, constant = dense_system.getEquation(equation_id)
        if verbose:
            print(f"  Updating solution based on equation {dense_system.equationToStr(equation_id)}")
        if dense_system.isIdentity(equation_id): 
            continue
        solution[first_vars[equation_id]] = np.bitwise_xor(constant, scalarProduct(equation, solution))
    
    return solution


def scalarProduct(bitarray1, bitarray2):
    # return the number of common 1's between two bitarrays modded by 2
    return bitarray.util.count_and(bitarray1, bitarray2) % 2


def test_simple_gaussian_elimination():
    matrix = [[0, 1, 2],
              [1, 2],
              [0, 2]]
    constants = [1, 0, 1]
    solution_str = "100"

    system = DenseModulo2System(solution_size=len(solution_str))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])

    assert gaussian_elimination(system, [2, 0, 1]).to01() == solution_str


def test_gaussian_with_swaps():
    matrix = [[1, 34],
              [0, 1, 34],
              [0, 34]]
    constants = [0, 1, 1]
    solution_str = "1" + "0" * 34

    system = DenseModulo2System(solution_size=len(solution_str))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])

    assert gaussian_elimination(system, [0, 1, 2]).to01() == solution_str


def test_empty_system():
    matrix = [[1, 34],
              [0, 1, 34],
              [0, 34]]
    constants = [0, 1, 1]
    solution_str = "0" * 35 
    system = DenseModulo2System(solution_size=len(solution_str))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])
    assert gaussian_elimination(system, []).to01() == solution_str


if __name__ == "__main__":
    test_simple_gaussian_elimination()
    test_gaussian_with_swaps()
    test_empty_system()
    
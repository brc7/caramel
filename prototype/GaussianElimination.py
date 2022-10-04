import numpy as np
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
                print(f"  Top Equation: Id: {top_equation_id}, Equation: "
                      f"{dense_system.equationToStr(top_equation_id)}")
                print(f"  Bot Equation: Id: {bot_equation_id}, Equation: "
                      f"{dense_system.equationToStr(bot_equation_id)}")

            if first_vars[top_equation_id] == first_vars[bot_equation_id]:
                dense_system.xorEquations(bot_equation_id, top_equation_id)
                if verbose:
                    print(f"    Top and Bot have equal first vars. Set Bot = Top"
                          f" XOR Bot. Bot becomes: "
                          f"{dense_system.equationToStr(bot_equation_id)}")
                if dense_system.isUnsolvable(top_equation_id):
                    raise UnsolvableSystemException(f"Equation {equation_id:d}"
                                f"has all coefficients = 0 but constant is 1.")
                #TODO if dense_system.isIdentity(top_equation_id) continue the outer for loop
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

    # TODO: how should we handle the solution?
    # Should this be a bitvector? Should we create it elsewhere and pass it in?
    solution = np.zeros(shape=dense_system.bitvector_size, dtype=dense_system.dtype)
    for i in range(len(relevant_equation_ids) - 1, -1, -1):
        equation_id = relevant_equation_ids[i]
        equation, constant = dense_system.getEquation(equation_id)
        if verbose:
            print(f"  Updating solution based on equation {dense_system.equationToStr(equation_id)}")
        if dense_system.isIdentity(equation_id): 
            continue
        if np.bitwise_xor(constant, scalarProduct(equation, solution)) == 1:
            solution = dense_system._update_bitvector(solution, first_vars[equation_id])
    
    return solution


def scalarProduct(array1, array2):
    # returns the bitwise and of two numpy arrays (not just the integer values 
    # but the number of 1s that overlap in both array1 and array2)
    # TODO: can we optimize this function?
    # TODO should we make a Modulo2Equation class/utils file?
    result = np.bitwise_and(array1, array2)
    sum = 0
    for number in result:
        if number != 0:
            sum += bin(number).count("1")
    return sum


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

    assert system.bitArrayToStr(gaussian_elimination(system, [2, 0, 1])) == solution_str


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

    assert system.bitArrayToStr(gaussian_elimination(system, [0, 1, 2])) == solution_str


if __name__ == "__main__":
    test_simple_gaussian_elimination()
    test_gaussian_with_swaps()
    
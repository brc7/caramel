import numpy as np
from Modulo2System import DenseModulo2System


#TODO: can we optimize this function?
#TODO should we make a Modulo2Equation class/utils file?
def scalarProduct(array1, array2):
    # returns the bitwise and of two numpy arrays (not just the integer values 
    # but the number of 1s that overlap in both array1 and array2)
    result = np.bitwise_and(array1, array2)
    sum = 0
    for number in result:
        if number != 0:
            sum += bin(number).count("1")
    return sum


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
                print(f"    Top Equation: Id: {top_equation_id}, Equation: "
                      f"{dense_system.equationToStr(top_equation_id)}")
                print(f"    Bot Equation: Id: {bot_equation_id}, Equation: "
                      f"{dense_system.equationToStr(bot_equation_id)}")

            if first_vars[top_equation_id] == first_vars[bot_equation_id]:
                dense_system.xorEquations(bot_equation_id, top_equation_id)
                if verbose:
                    print(f"        First vars of both equations equal, after "
                          f"XOR bot equation is: "
                          f"{dense_system.equationToStr(bot_equation_id)}")
                if dense_system.isUnsolvable(top_equation_id):
                    raise ValueError("Unsolvable System")
                #TODO if dense_system.isIdentity(top_equation_id) continue the outer for loop
                first_vars[bot_equation_id] = dense_system.getFirstVar(bot_equation_id)

            if first_vars[top_equation_id] > first_vars[bot_equation_id]:
                if verbose:
                    print(f"        Swapping equations based on first vars.")
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
            print(f"    Updating solution based on equation {dense_system.equationToStr(equation_id)}")
        if dense_system.isIdentity(equation_id): 
            continue
        if np.bitwise_xor(constant, scalarProduct(equation, solution)) == 1:
            solution = dense_system.update_bitvector(solution, first_vars[equation_id])
    
    return solution


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

    assert system.bitArrayToStr(gaussian_elimination(system, [0, 1, 2])) == solution_str

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
    
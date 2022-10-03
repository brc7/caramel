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


def gaussian_elimination(dense_system, relevant_equation_ids, verbose=False):
    first_vars = {}
    for equation_id in relevant_equation_ids:
        first_vars[equation_id] = dense_system.getFirstVar(equation_id)
    
    for i in range(len(relevant_equation_ids) - 1):
        for j in range(i + 1, len(relevant_equation_ids)):
            eqI = relevant_equation_ids[i]
            eqJ = relevant_equation_ids[j]

            if verbose:
                print(f"\nStarting Iteration looking at equations {eqI} and {eqJ}")
                print(f"\nStarting Iteration: I:{bin(dense_system.getEquation(eqI)[0][0])} and J:{bin(dense_system.getEquation(eqJ)[0][0])}")

            if first_vars[eqI] == first_vars[eqJ]:
                print("First vars equal, XOR the second with the first and replace the second")
                dense_system.xorEquations(eqJ, eqI)
                print(f"After Xor: I:{bin(dense_system.getEquation(eqI)[0][0])} and J:{bin(dense_system.getEquation(eqJ)[0][0])}")

                if dense_system.isUnsolvable(eqI):
                    raise ValueError("Unsolvable System")
                #TODO if dense_system.isIdentity(eqI) continue the outer for loop
                first_vars[eqJ] = dense_system.getFirstVar(eqJ)
                print(f"new first var of J: {first_vars[eqJ]}")

            if first_vars[eqJ] > first_vars[eqI]:
                print(f"swapping equations: I:{bin(dense_system.getEquation(eqI)[0][0])} and J:{bin(dense_system.getEquation(eqJ)[0][0])}")
                dense_system.swapEquations(eqI, eqJ)
    
    solution = np.zeros(shape=dense_system.bitvector_size, dtype=dense_system.dtype)
    for i in range(len(relevant_equation_ids) - 1, -1, -1):
        equation_id = relevant_equation_ids[i]
        equation, constant = dense_system.getEquation(equation_id)
        print(f"i: {i}, equation_id: {equation_id}, equation: {bin(equation[0])}")
        if dense_system.isIdentity(equation_id): 
            continue
        if np.bitwise_xor(constant, scalarProduct(equation, solution)) == 1:
            solution = dense_system.update_bitvector(solution, first_vars[equation_id])
    
    return solution


def test_gaussian_elimination():
    matrix = [[0, 1, 2],
              [1, 2],
              [0, 2]]
    constants = [1, 0, 1]
    solution = [1, 0, 0]

    system = DenseModulo2System(solution_size=len(solution))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])

    assert bin(gaussian_elimination(system, [0, 1, 2])[0]) == "0b1"


def test_gaussian_elimination1():
    matrix = [[1, 2],
              [0, 1, 2],
              [0, 2]]
    constants = [0, 1, 1]
    solution = [1, 0, 0]

    system = DenseModulo2System(solution_size=len(solution))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])

    assert bin(gaussian_elimination(system, [0, 1, 2])[0]) == "0b1"


if __name__ == "__main__":
    test_gaussian_elimination()
    test_gaussian_elimination1()
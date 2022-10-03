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


def gaussian_elimination(dense_system, relevant_equation_ids):
    first_vars = {}
    for equation_id in relevant_equation_ids:
        first_vars[equation_id] = dense_system.getFirstVar(equation_id)
    
    for i in range(len(relevant_equation_ids) - 1):
        for j in range(i + 1, len(relevant_equation_ids)):
            eqJ = relevant_equation_ids[j]
            eqI = relevant_equation_ids[i]

            if first_vars[eqI] == first_vars[eqJ]:
                dense_system.xorEquations(eqI, eqJ)
                if dense_system.isUnsolvable(eqI):
                    raise ValueError("Unsolvable System")
                #TODO if dense_system.isIdentity(eqI) continue the outer for loop
                first_vars[eqI] = dense_system.getFirstVar(eqI)

            if first_vars[eqI] > first_vars[eqJ]:
                dense_system.swap(eqI, eqJ)
    
    # TODO should we put the solution in the system itself?
    solution = np.zeros(shape=dense_system.bitvector_size, dtype=dense_system.dtype)
    for i in range(len(relevant_equation_ids), -1, -1):
        equation_id = relevant_equation_ids[i]
        if dense_system.isIdentity(equation_id): 
            continue
        equation, constant = dense_system.getEquation(equation_id)
        solution[first_vars[equation_id]] = np.bitwise_xor(constant, scalarProduct(equation, solution))
    
    return solution


def test_gaussian_elimination():
    matrix = [[1, 1, 1],
              [0, 1, 1],
              [1, 0, 1]]
    constants = [1, 0, 1]
    solution = [1, 0, 0]

    system = DenseModulo2System(solution_size=len(solution))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])

    print(gaussian_elimination(system, [0, 1, 2]))


# def test_gaussian_elimination1():
#     matrix = [[0, 0, 1],
#               [1, 1, 1]]
#     constants = [1, 0]
#     solution = [1, 0, 1]

#     system = DenseModulo2System(solution_size=len(solution))
#     for i, vars in enumerate(matrix):
#         system.addEquation(equation_id=i, 
#                            participating_variables=vars, 
#                            constant=constants[i])

#     print(regular_gaussian_elimination(system, [0, 1, 2]))


from prototype.Modulo2System import DenseModulo2System
from caramel import regular_gaussian_elimination

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

    print(regular_gaussian_elimination(system, [0, 1, 2]))


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
    
    
    
test_gaussian_elimination()
# test_gaussian_elimination1()
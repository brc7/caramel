from caramel.Modulo2System import DenseModulo2System
from caramel.GaussianElimination import gaussian_elimination


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
from prototype.Modulo2System import DenseModulo2System

def test_gaussian_elimination():
    matrix = [[1, 0, 1],
              [0, 1, 0],
              [0, 0, 1]]
    constants = [1, 0, 1]
    solution = [0, 0, 1]

    system = DenseModulo2System(solution_size=len(solution))
    for i, vars in enumerate(matrix):
        system.addEquation(equation_id=i, 
                           participating_variables=vars, 
                           constant=constants[i])

    
    

test_gaussian_elimination()
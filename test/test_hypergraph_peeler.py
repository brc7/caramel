import random
from collections import defaultdict
from caramel.HypergraphPeeler import peel_hypergraph
from caramel.Modulo2System import SparseModulo2System

def peel_random_hypergraph(num_equations, num_variables, verbose=0):
    sparse_system = SparseModulo2System(num_variables)
    equation_ids = list(range(num_equations))
    for equation_id in equation_ids:
        variables = [random.randrange(0, num_variables) for _ in range(3)]
        constant = random.choice([0,1])
        sparse_system.addEquation(equation_id, variables, constant)

    unpeeled, peeled, order, system = peel_hypergraph(
        sparse_system, equation_ids, verbose=verbose)

    success = verify_peeling_order(
        unpeeled, peeled, order, sparse_system, equation_ids, verbose=verbose)
    return success


def verify_peeling_order(unpeeled, peeled, order, system,
    equation_ids, verbose=0):
    # Verify that unpeeled and peeled are disjoint (equation cannot be both).
    intersection = set(unpeeled).intersection(set(peeled))
    if len(intersection):
        if verbose:
            print(f"The following equation_ids are both peeled and unpeeled: "
                  f"{intersection}")
        return False
    # Verify that all equation_ids are either peeled or unpeeled.
    difference = set(equation_ids).difference(set(unpeeled + peeled))
    if len(difference) != 0:
        if verbose:
            print(f"The following equation_ids are neither peeled nor "
                  f"unpeeled: {difference}")
        return False

    # Reverse the solution order to get the peeling order.
    order.reverse()
    peeled.reverse()

    # First, create a graph to check against.
    variable_to_equations = defaultdict(list)
    for equation_id in equation_ids:
        participating_variables, _ = system.getEquation(equation_id)
        for variable in participating_variables:
            variable_to_equations[variable].append(equation_id)

    # Now we verify the equation peeling order by actually peeling the graph.
    for variable_id, equation_id in zip(order, peeled):
        # 1. Hinge variable should participate in only one unpeeled equation.
        participating_equations = variable_to_equations[variable_id]
        if len(participating_equations) != 1:
            if verbose:
                print(f"{variable_id} participates in more than one equation:"
                      f"{participating_equations}")
            return False
        # 2. The equation should be equation_id
        if participating_equations[0] != equation_id:
            if verbose:
                print(f"{variable_id} participates in equation: "
                      f"{participating_equations[0]} not {equation_id}")
            return False
        # 3. Remove the peeled equation from the graph
        participating_variables, _ = system.getEquation(equation_id)
        for participating_variable in participating_variables:
            variable_to_equations[participating_variable].remove(equation_id)

    # Verify that no other equations could be peeled, but weren't.
    for equation_id in unpeeled:
        participating_variables, _ = system.getEquation(equation_id)
        for participating_variable in participating_variables:
            num_equations = len(variable_to_equations[participating_variable])
            if num_equations < 2:
                if verbose:
                    print(f"Equation {equation_id} can be peeled via variable "
                        f"{participating_variable}, which is in equations: "
                        f"{variable_to_equations[participating_variable]}")
                return False
    return True

def test_random_inputs():
    hypergraph_dimensions = [(random.randint(10,100),random.randint(10,100))
                             for _ in range(10000)]

    for n, m in hypergraph_dimensions:
        success = peel_random_hypergraph(n, m, verbose=0)
        # if not success:
        #     print(f"[{n:d},{m:d}]: {success}")
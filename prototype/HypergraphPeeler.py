import random
import numpy as np
from collections import defaultdict 
from Modulo2System import SparseModulo2System

'''
We implement the hypergraph peeling method of "Cache-Oblivious Peeling of
Random Hypergraphs" by Belazzougui1, Boldi, Ottaviano, Venturini, and Vigna
(https://arxiv.org/pdf/1312.0526.pdf). The method described in the paper is
a slightly more complicated version of the one that is actually used in
practice and in the reference implementation. Also, note that in the context
of linear systems, the term "edge" refers to "equation" and "vertex" refers to
"variable". We use the simpler method, which proceeds as follows:

Given a hypergraph, we begin by scanning the hypergraph for a degree-1 variable
(that is, a variable that appears in exactly one equation). We peel this
equation from the system, and observe that this removal could have "freed up"
other edges. We can identify candidate equations to peel by checking the other
variables from the equation we just removed - if one of these variables now
appears in exactly 1 equation, we have found our next equation to peel. We
recursively repeat this process until we cannot peel any more edges.
'''

def peel_hypergraph(sparse_system, equation_ids, verbose=0):
    num_equations, num_variables = sparse_system.shape
    if verbose >= 1:
        print(f"Peeling hypergraph of ({num_variables:d} vertices (variables) "
              f"and {num_equations:d} edges (equations). ")

    # The paper uses "edge" for equation and "vertex" for variable.

    # Degree of a variable is the number of unpeeled equations that contain it.
    degree = np.zeros(num_variables, dtype=int)
    # equation_is_peeled[equation_id] = 1 if variable_id has been peeled.
    equation_is_peeled = np.zeros(num_equations, dtype=int)
    # Stores the XOR of the edges (equations) each variable participates in.
    equation_id_xors = np.zeros(num_variables, dtype=int)

    for equation_id in equation_ids:
        participating_variables, _ = sparse_system.getEquation(equation_id)
        if verbose >= 2:
            print(f"Edge (equation) {equation_id}: "
                  f"<{','.join(str(v) for v in participating_variables)}>")
        for variable_id in participating_variables:
            # Increment the degree for each vertex in the incident edge.
            degree[variable_id] += 1
            # Add the edge to the xor list of the variable_id.
            equation_id_xors[variable_id] = np.bitwise_xor(
                equation_id_xors[variable_id], equation_id)

    # This is a stack of variables, in reverse peeling order.
    vertex_stack = []
    stack_top = 0

    for variable_id in range(num_variables):
        if degree[variable_id] == 1:
            # Then we should peel the only equation containing variable_id.
            vars_to_peel = [variable_id]
            num_processed = 0
            while num_processed < len(vars_to_peel):
                # The first trip through this loop, we peel the equation
                # containing variable_id. Subsequent trips peel equations that
                # have become "freed up" by previous peeling steps.
                var_to_peel = vars_to_peel[num_processed]
                num_processed += 1
                # If degree is zero, then we've already peeled this equation.
                if degree[var_to_peel] != 1:
                    continue
                vertex_stack.append(var_to_peel)
                # Because var_to_peel participates in only one unpeeled
                # equation, equation_id_xors contains that equation_id.
                peeled_equation_id = equation_id_xors[var_to_peel]
                equation_is_peeled[peeled_equation_id] = 1
                if verbose >= 2:
                    print(f'Peeling variable {var_to_peel} in equation {peeled_equation_id}')
                    print(f'Peeling order: {vertex_stack}')
                # We must remove peeled_equation_id from equation_id_xors for
                # the other variables that participate in this equation.
                vars_to_update, _ = sparse_system.getEquation(
                    peeled_equation_id)
                for var_to_update in vars_to_update:
                    # Since we peeled the equation, decrease the degree.
                    degree[var_to_update] -= 1
                    if verbose >= 3:
                        print(f"Removing equation {peeled_equation_id} from {var_to_update}. Degree[{var_to_update}] = {degree[var_to_update]}. "
                              f"Old XOR_list[{var_to_update}] = {equation_id_xors[var_to_update]}. "
                              f"New XOR_list[{var_to_update}] = {np.bitwise_xor(equation_id_xors[var_to_update], peeled_equation_id)}. ")

                    if var_to_update != var_to_peel:
                        equation_id_xors[var_to_update] = np.bitwise_xor(
                            equation_id_xors[var_to_update], peeled_equation_id)
                # Because of how the hashing construction works, vars_to_update
                # might contain duplicates. We de-dupe to process it only once.
                for var_to_maybe_peel in set(vars_to_update):
                    if degree[var_to_maybe_peel] == 1:
                        vars_to_peel.append(var_to_maybe_peel)

    unpeeled_equation_ids = []
    for equation_id in equation_ids:
        if equation_is_peeled[equation_id] == 0:
            unpeeled_equation_ids.append(equation_id)

    # Variables, listed in the order that they should be solved via back-sub.
    vertex_stack.reverse()
    var_solution_order = vertex_stack
    # This array is ordered so that peeled_equation_ids[n] can be used to solve
    # for the variable with variable_id = var_solution_order[n].
    peeled_equation_ids = [equation_id_xors[var] for var in var_solution_order]

    if verbose >= 1:
        print(f"Peeled {len(peeled_equation_ids):d} of {num_equations:d}: "
              f"({100*len(peeled_equation_ids)/num_equations:.2f}%)")
    
    state = (unpeeled_equation_ids, peeled_equation_ids,
             var_solution_order, sparse_system)
    return state


def test_random_hypergraph(num_equations, num_variables, verbose=0):
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


if __name__ == '__main__':
    hypergraph_dimensions = [(random.randint(10,100),random.randint(10,100))
                             for _ in range(10000)]

    for n, m in hypergraph_dimensions:
        success = test_random_hypergraph(n, m, verbose=0)
        if not success:
            print(f"[{n:d},{m:d}]: {success}")

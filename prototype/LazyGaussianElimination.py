import numpy as np
import math
from collections import defaultdict
from Modulo2System import *

'''
This implements the lazy gaussian elimination algorithm of Genuzio, Ottaviano
and Vigna from Fast Scalable Construction of (Minimal Perfect Hash) Functions
(https://arxiv.org/pdf/1603.04330.pdf). We consider a system of equations where
a variable can be either {active, idle, solved} and an equation can be
{sparse, dense}. The intuition is that the values of active variables must be
explicitly solved for (typically via gaussian elimination). The values of the
solved variables can be easily computed from the active variables. Idle
variables and sparse equations are those that we have not yet tried to solve. 
The algorithm runs until all equations are dense and no variables are idle,
while preserving the following invariants:
1. Dense equations do not contain idle variables
2. An equation can contain at most one solved variable
3. A solved variable appears in exactly one dense equation

The algorithm attempts to maximize the "usefulness" of each active variable, by
greedily activating idle variables that appear in many unsolved equations (and
can thus be used to solve for the largest possible number of other variables).
This is done via priority queues, where the weight of a variable is the number
of sparse (unsolved) equations where it appears and the priority of an equation
is the number of idle variables it contains. We iterate the following steps:
- If there is a non-empty sparse equation of priority zero, make it dense.
- If there is sparse equation of priority one, it has only one idle variable.
    This variable becomes solved, this equation becomes dense, and we do
    single steps of Gaussian elimination to remove the variable from all other
    equations.
- Otherwise, the idle variable appearing in the largest number of sparse
    equations becomes active.

Once this process completes, the solution to the active variables can be found
by standard Gaussian elimination on the dense equations that do not contain
solved variables. The solved variables can be computed using the dense
equations that do contain solved variables.
'''


def lazy_gaussian_elimination(sparse_system, equation_ids, verbose = 0):
    # Performs lazy gaussian elimination on a subset of the linear system
    # specified by sparse_system.
    # 
    # Parameters:
    #   sparse_system: A fully-specified SparseModulo2System. We assume that
    #       the equation_ids of the system range are integers in the range
    #       [0, num_equations-1] and the variable_ids are integers in
    #       the range [0, num_variables-1].
    #   equation_ids: The equations that should be considered during the
    #       lazy gaussian elimination process (e.g. the equations that remain
    #       after hypergraph peeling).
    #   verbose: Integer. Logging level - 0 prints nothing, 1 prints aggregate
    #       information and 2 prints iteration-level information.
    # 
    # Returns:
    #   A DenseModulo2System containing the active sub-system found by lazy
    #   gaussian elimination.
    num_equations, num_variables = sparse_system.shape
    var_to_equations, equation_priority, variable_weight, dense_system = \
        construct_dense_system(sparse_system, equation_ids)
    
    if verbose >= 2:
        print(f"System ({num_equations} equations, {num_variables} variables)")
    if verbose >= 3:
        print(dense_system.systemToStr())

    # List of sparse equations with priority 0 or 1. Probably needs a re-name.
    sparse_equation_ids = [equation_id for equation_id in equation_ids
                           if equation_priority[equation_id] <= 1]
    # List of dense equations with entirely active variables.
    dense_equation_ids = []
    # Equations that define a solved variable in terms of active variables.
    solved_equation_ids = []
    solved_variable_ids = []
    # List of currently-idle variables. Starts as a bit vector of all 1's, and
    # is filled in with 0s as variables become non-idle.
    
    # Ugly and breaks encapsulation, but there truly seems to be no other way.
    num_variables_per_chunk = dense_system.dtype.itemsize * 8
    idle_variable_indicator = -1 * np.ones(
        int(math.ceil(num_variables / num_variables_per_chunk)),
        dtype=dense_system.dtype)

    # Sorted list of variable ids, in descending weight order.
    sorted_variable_ids = countsort_variable_ids(variable_weight,
                                                 num_variables,
                                                 num_equations)

    num_active_variables = 0
    num_remaining_equations = num_equations

    while(num_remaining_equations > 0):
        if verbose >= 3:
            print(f"Solved variable IDs ({len(solved_variable_ids)} variables): ", " ".join([f"{v}" for v in sorted(solved_variable_ids)]))
            print(f"Idle variable IDs ({len(sorted_variable_ids)} variables):", " ".join([f"{v}({variable_weight[v]})" for v in sorted(sorted_variable_ids)]))
            # print(f"Active variable IDs ({len(active_variable_ids)} variables):", " ".join([f"{v}" for v in sorted(active_variable_ids)]))
            print(f"Solved equations ({len(solved_equation_ids)} equations): ", solved_equation_ids)
            print(f"Active equations ({len(dense_equation_ids)} equations): ", dense_equation_ids)
            print("System: ")
            print(dense_system.systemToStr())
        if not sparse_equation_ids:
            # If there are no sparse equations with priority 0 or 1, then
            # we make another variable active and see if this status changes.
            variable_id = sorted_variable_ids.pop()
            while not variable_weight[variable_id]:
                # Skip variables with weight = 0 (these are already solved).
                variable_id = sorted_variable_ids.pop()
            if verbose >= 2:
                print(f"Making variable {variable_id:d} with weight "
                      f"{variable_weight[variable_id]:d} active "
                      f"({num_remaining_equations:d} equations remaining).")
            # Mark variable as no longer idle. This is ugly and breaks
            # encapsulation, but there truly seems to be no other way.
            dense_system._update_bitvector(idle_variable_indicator,
                                           variable_id,
                                           value=0)
            # active_variable_ids.append(variable_id)
            num_active_variables += 1
            # By marking this variable as active, we must update priorities.
            for equation_id in var_to_equations[variable_id]:
                equation_priority[equation_id] -= 1
                # TODO: LOOK AT THIS
                if equation_priority[equation_id] == 1:
                    sparse_equation_ids.append(equation_id)
        else:
            # There is at least one sparse equation with priority 0 or 1.
            num_remaining_equations -= 1
            equation_id = sparse_equation_ids.pop()
            if verbose >= 2:
                print(f"Equation {equation_id:d} is sparse with priority "
                        f"{equation_priority[equation_id]:d}. "
                        f"{num_remaining_equations} equations remain.")
            if equation_priority[equation_id] == 0:
                if verbose >= 2:
                    print(f"Equation {equation_id:d} has no idle variables.")
                equation, constant = dense_system.getEquation(equation_id)
                equation_is_nonempty = np.sum(equation)
                if equation_is_nonempty:
                    # Since priority is 0, all variables are active.
                    dense_equation_ids.append(equation_id)
                elif constant != 0:  # The equation is unsolvable.
                    raise UnsolvableSystemException(f"Equation {equation_id:d}"
                                     f"has all coefficients = "
                                     f"0 but constant is 1.")
                else:
                    if verbose >= 2:
                        print(f"Equation {equation_id:d} is an identity.")
                # The remaining case corresponds to an identity equation
                # (which is empty, but so is the output so it's fine).
            elif equation_priority[equation_id] == 1:
                # If there is only 1 idle variable, the equation is solved.
                # We need to find the pivot - the variable_id of the only
                # remaining idle variable in the equation.
                equation, constant = dense_system.getEquation(equation_id)
                for chunk_id, chunk in enumerate(equation):
                    flag = np.bitwise_and(
                        chunk, idle_variable_indicator[chunk_id])
                    if flag > 0:
                        break
                # Performance note: flag is a power of 2, so the log2 can be
                # done via bit shifting.
                variable_id = chunk_id * num_variables_per_chunk
                variable_id += int(np.log2(flag))
                if verbose >= 2:
                    print(f"Equation {equation_id:d} is solved by variable "
                          f"{variable_id}.")
                solved_variable_ids.append(variable_id)
                solved_equation_ids.append(equation_id)
                # By making the weight 0, we will skip this variable_id in the 
                # future when looking for new active variables.
                variable_weight[variable_id] = 0
                # Remove this variable from all other equations.
                for other_equation_id in var_to_equations[variable_id]:
                    if other_equation_id != equation_id:
                        equation_priority[other_equation_id] -= 1
                        if equation_priority[other_equation_id] == 1:
                            sparse_equation_ids.append(other_equation_id)
                        if verbose >= 2:
                            print(f"Adding equation {equation_id:d} to "
                                  f"equation {other_equation_id:d}.")
                        # Perform one step of gaussian elimination.
                        dense_system.xorEquations(other_equation_id,
                                                  equation_id)
    if verbose >= 1:
        print(f"{num_active_variables:d} active of {num_variables:d} "
              f"total variables ({100*num_active_variables/num_variables:.2f}%).")
    if verbose >= 2:
        print(f"Dense equations: {dense_equation_ids}")
        print(f"Solved equations: {solved_equation_ids}")
        print(f"Solved variables: {solved_variable_ids}")

    state = (dense_equation_ids, solved_equation_ids,
             solved_variable_ids, dense_system)
    return state


def construct_dense_system(sparse_system, equation_ids):
    num_equations, num_variables = sparse_system.shape
    # The weight is the number of sparse equations containing variable_id.
    variable_weight = np.zeros(shape=num_variables, dtype=int)
    # The equation priority is the number of idle variables in equation_id.
    equation_priority = np.zeros(shape=num_equations, dtype=int)

    dense_system = DenseModulo2System(num_variables)
    var_to_equations = defaultdict(list)
    for equation_id in equation_ids:
        participating_vars, constant = sparse_system.getEquation(equation_id)
        # We should only add a variable to the equation in the dense system if
        # it appears an odd number of times. This is because we compute output
        # as XOR(solution[hash_1], solution[hash_2] ...). If hash_1 = hash_2 = 
        # variable_id, then XOR(solution[hash_1], solution[hash_2]) = 0 and
        # the variable_id did not actually participate in the solution.
        vars_to_add = set()
        for variable_id in participating_vars:
            if variable_id not in vars_to_add:
                vars_to_add.add(variable_id)
            else:
                vars_to_add.remove(variable_id)
        # Update weight and priority for de-duped variables.
        dense_system.addEquation(equation_id, list(vars_to_add), constant)
        for variable_id in vars_to_add:
            variable_weight[variable_id] += 1
            equation_priority[equation_id] += 1
            var_to_equations[variable_id].append(equation_id)
    return var_to_equations, equation_priority, variable_weight, dense_system


def countsort_variable_ids(variable_weight, num_variables, num_equations):
    # Sorts variables in ascending weight order in O(num_variables + max_weight) time.
    sorted_variable_ids = list(range(num_variables))
    counts = np.zeros(shape=num_equations+1, dtype=int)
    for variable_id in range(num_variables):
        counts[variable_weight[variable_id]] += 1
    counts = np.cumsum(counts)
    for variable_id in reversed(range(num_variables)):
        count_idx = variable_weight[variable_id]
        counts[count_idx] -= 1
        sorted_variable_ids[counts[count_idx]] = variable_id
    return sorted_variable_ids

def test_random_system(num_equations, num_variables, verbose=0):
    sparse_system = SparseModulo2System(num_variables)
    equation_ids = list(range(num_equations))
    if verbose >= 1:
        print('Constructing linear system...')
    for equation_id in equation_ids:
        variables = [random.randrange(0, num_variables) for _ in range(3)]
        constant = random.choice([0,1])
        sparse_system.addEquation(equation_id, variables, constant)
    if verbose >= 1:
        print('Solving linear system...')
    try:
        state = lazy_gaussian_elimination(sparse_system,
                                        equation_ids,
                                        verbose=verbose)
    except UnsolvableSystemException as e:
        return True
    return True


def test_unsolvable_pair(verbose=0):
    # Tests the case where two equations have incompatible constants.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [5,8,9], 1)  # Unsolvable duplicate equation.
    equation_ids = [0,1,2,3,4,5]
    try: 
        _ = lazy_gaussian_elimination(sparse_system,
                                      equation_ids,
                                      verbose=verbose)
    except UnsolvableSystemException as e:
        return True
    return False

def test_solvable_system(verbose=0):
    # Tests a system that is known to be solvable.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    equation_ids = [0,1,2,3,4,5]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True


def test_active_system(verbose=0):
    # Tests a system that is known to be solvable, but has an active core.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    sparse_system.addEquation(6, [2,8,9], 1)
    sparse_system.addEquation(7, [0,7,9], 1)
    sparse_system.addEquation(8, [1,7,9], 0)
    sparse_system.addEquation(9, [1,2,9], 0)
    equation_ids = [0,1,2,3,4,5,6,7,8,9]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True

def test_active_with_duplicate_system(verbose=0):
    # Tests a system that is known to be solvable, but has an active core.
    num_variables = 10
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [1,2,3], 1)
    sparse_system.addEquation(1, [3,4,5], 1)
    sparse_system.addEquation(2, [4,5,6], 0)
    sparse_system.addEquation(3, [6,7,8], 1)
    sparse_system.addEquation(4, [5,8,9], 0)
    sparse_system.addEquation(5, [0,8,9], 1)
    sparse_system.addEquation(6, [2,8,9], 1)
    sparse_system.addEquation(7, [0,7,9], 1)
    sparse_system.addEquation(8, [1,7,9], 0)
    sparse_system.addEquation(9, [1,2,9], 0)
    equation_ids = [0,1,2,3,4,5,6,7,8,9]
    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True

def test_solvable_with_duplicates_system(verbose=0):
    # Tests a system that is known to be solvable.
    num_variables = 9
    sparse_system = SparseModulo2System(num_variables)
    sparse_system.addEquation(0, [0,1,2], 1)
    sparse_system.addEquation(1, [2,3,4], 1)
    sparse_system.addEquation(2, [3,4,5], 0)
    sparse_system.addEquation(3, [5,6,7], 1)
    sparse_system.addEquation(4, [0,8,8], 1)
    equation_ids = [0,1,2,3,4]

    try: 
        state = lazy_gaussian_elimination(sparse_system,
                                          equation_ids,
                                          verbose=verbose)
    except UnsolvableSystemException as e:
        return False
    return True


if __name__ == '__main__':
    test_unsolvable_pair(verbose=2)
    test_solvable_system(verbose=2)
    test_active_system(verbose=2)
    test_solvable_with_duplicates_system(verbose=2)
    test_active_with_duplicate_system(verbose=2)

    hypergraph_dimensions = [(random.randint(10,100),random.randint(10,100))
                            for _ in range(1000)]

    for n, m in hypergraph_dimensions:
        success = test_random_system(n, m, verbose=0)

    # N = 100000
    # M = int(1.14 * N)
    # success = test_random_system(N, M, verbose=1)

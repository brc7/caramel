import math
import random

import spookyhash
from caramel.BackSubstitution import (solve_lazy_from_dense,
                                      solve_peeled_from_dense, sparse_to_dense,
                                      scalarProduct)
from caramel.BucketedHashStore import BucketedHashStore
from caramel.CSF import CSF
from caramel.Codec import canonical_huffman
from caramel.GaussianElimination import gaussian_elimination
from caramel.HypergraphPeeler import peel_hypergraph
from caramel.LazyGaussianElimination import lazy_gaussian_elimination
from caramel.Modulo2System import (DenseModulo2System, SparseModulo2System,
                                   UnsolvableSystemException)


def construct_modulo2_system(key_hashes, values, codedict, seed, verbose=0):
    """
    Constructs a binary system of linear equations to solve for each bit of the 
    encoded values for each key.

	Arguments:
		key_hashes: An iterable collection of N unique hashes.
        values: An interable collection of N values corresponding to key_hashes
		encoded_values: An iterable collection of N bitarrays, representing the 
            encoded values. 
        seed: A seed for hashing. If callling construct_modulo2_system again 
            after a failed solve attempt, should increment this seed by 3.

	Returns:
		SparseModulo2System to solve for each key's encoded bits.
    """

    # This is a constant multiplier on the number of variables based on the 
    # number of equations expected. This constant makes the system solvable with
    # very high probability. If we want faster construction at the cost of 12%
    # more memory, we can omit lazy gaussian elimination and set delta to 1.23
    DELTA = 1.10 

    num_equations = sum(len(codedict[v]) for v in values)
    num_variables = math.ceil(num_equations * DELTA)

    if verbose >= 1:
        print(f"Constructing SparseModulo2System with {num_variables} variables "
              f"and {num_equations} equations")

    sparse_system = SparseModulo2System(num_variables)
    
    equation_id = 0
    for i, key_hash in enumerate(key_hashes):
        start_var_locations = []
        temp_seed = seed
        for _ in range(3):
            #TODO lets do sebastiano's trick here instead of modding
            #TODO lets write a custom hash function that generates three 64 bit 
            # hashes instead of hashing 3 times
            start_var_locations.append(
                spookyhash.hash64(int.to_bytes(key_hash, 64, "big"), temp_seed) % num_variables
            )
            temp_seed += 1
                
        if verbose >= 2:
            print(f"  Constructing Equations for value: {values[i]} with code {codedict[values[i]]}")

        for offset, bit in enumerate(codedict[values[i]]):
            participating_variables = []
            for start_var_location in start_var_locations:
                var_location = start_var_location + offset
                if var_location >= num_variables:
                    var_location -= num_variables
                participating_variables.append(var_location)
            
            if verbose >= 2:
                print(f"    id: {equation_id}, vars: {participating_variables}, bit: {bit}")

            sparse_system.addEquation(equation_id, participating_variables, bit)
            equation_id += 1
            
    return sparse_system


def solve_modulo2_system(sparse_system, verbose=0):
    # Throws UnsolvableSystemException
    # 1. Do hypergraph peeling on the sparse system to remove easy equations.
    if verbose >= 1:
        print(f"Attempting to solve system of {sparse_system.shape[0]} equations"
              f" and {sparse_system.shape[1]} variables.")
        if verbose >= 3:
            dense_system = sparse_to_dense(sparse_system)
            print(f"System: ")
            print(dense_system.systemToStr())
    state = peel_hypergraph(sparse_system,
                            sparse_system.equation_ids,
                            verbose=verbose-1)
    unpeeled_ids, peeled_ids, var_order, sparse_system = state
    if verbose >= 1:
        print(f"Peeled {len(peeled_ids)} equations. ("
              f"{100 * len(peeled_ids) / sparse_system.shape[0]:.2f}% of total)")
    # 2. Do lazy elimination on the system to short-cut gaussian elimination.
    state = lazy_gaussian_elimination(sparse_system,
                                      unpeeled_ids,
                                      verbose=verbose-1)
    dense_ids, solved_ids, solved_vars, dense_system = state
    # # Un-performant hack (large memory)
    # dense_system = sparse_to_dense(sparse_system)
    if verbose >= 1:
        print(f"Lazily solved {len(solved_ids)} equations. ("
              f"{100 * len(solved_ids) / sparse_system.shape[0]:.2f}% of total)")
    # 3. Do regular gaussian elimination on whatever is left.
    if verbose >= 1:
        print(f"Solving {len(dense_ids)} equations via Gaussian Elimination. ("
              f"{100 * len(dense_ids) / sparse_system.shape[0]:.2f}% of total)")
    solution = gaussian_elimination(dense_system,
                                    dense_ids,
                                    verbose=verbose-1)
    # 4. Back-substitute the lazily solved variables.
    solution = solve_lazy_from_dense(
        dense_ids, solved_ids, solved_vars, dense_system, solution)
    # 5. Back-substitute the peeled variables.
    solution = solve_peeled_from_dense(
        peeled_ids, var_order, sparse_system, solution)
    # 6. Done. We can return the dense solution

    # Check the solution
    original_system = sparse_to_dense(sparse_system)
    # Do the inner products explicitly for each row
    for equation_id in original_system.equation_ids:
        equation, constant = original_system.getEquation(equation_id)
        result = scalarProduct(equation, solution) % 2
        print(f"Equation {equation_id}: {result}, {constant}")

    return solution


def construct_csf(keys, values, verbose=0):
    """
    Constructs a compressed static function. This implementation sacrifices a 
    lot of the configurability for the sake of simplicity.

	Arguments:
		keys: An iterable collection of N unique hashable items.
		values: An iterable collection of N values. 

	Returns:
		A csf structure supporting the .query(key) method.
    """

    # The code dict needs to be the same for all the partition-CSFs.
    codedict, code_length_counts, symbols = canonical_huffman(values, verbose=verbose)

    vectorizer = lambda s : bytes(s, 'utf-8')
    hash_store = BucketedHashStore(vectorizer, keys, values)
    if verbose >= 1:
        print(f"Divided keys into {len(list(hash_store.buckets()))} buckets")


    construction_seeds = []
    solutions = []
    solution_sizes = []
    for key_hashes, values in hash_store.buckets():
        if verbose >= 1:
            print(f"Solving system for {len(key_hashes)} key-value pairs.")
        seed = 0
        max_num_attempts = 0
        num_tries = 10
        while True:
            try:
                sparse_system = construct_modulo2_system(key_hashes,
                                                         values,
                                                         codedict,
                                                         seed,
                                                         verbose=verbose)
                solution = solve_modulo2_system(sparse_system, verbose=verbose)
                solutions.append(solution)
                solution_sizes.append(sparse_system.shape[1])
                construction_seeds.append(seed)
                break
            except UnsolvableSystemException as e:
                # system construction increments the seed and hashes 3 times, 
                # we add 3 here to ensure hashes are unique across runs
                seed += 3
                max_num_attempts += 1

                if max_num_attempts == num_tries:
                    raise ValueError(f"Attempted to solve system {num_tries} "
                                     f"times without success.")

    return CSF(vectorizer, hash_store.seed, solutions, solution_sizes, construction_seeds, symbols, code_length_counts)


if __name__ == '__main__':
    keys = ["key_1", "key_2", "key_3", "key_4", "key_5"]
    values = [111, 222, 333, 444, 555]
    try:
        csf = construct_csf(keys, values, verbose=0)
    except:
        pass

    keys = [str(i) for i in range(10)]
    random.seed(41)
    values = [math.floor(math.log(random.randint(1, 100)))
              for _ in range(len(keys))]

    print(keys)
    print(values)

    csf = construct_csf(keys, values, verbose=10)

    for key, value in zip(keys, values):
        print(csf.query(key), value)
        # assert csf.query(key) == value

    # for key, value in zip(keys, values):
    #     assert csf.query(key) == value

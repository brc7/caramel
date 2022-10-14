import math
import random
import time
from multiprocessing import Pool
import spookyhash
import numpy as np
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
    return solution


def construct_csf_for_bucket(key_hashes, values, codedict, verbose=0):
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
            return solution, seed
        except UnsolvableSystemException as e:
            # system construction increments the seed and hashes 3 times, 
            # we add 3 here to ensure hashes are unique across runs
            seed += 3
            max_num_attempts += 1

            if max_num_attempts == num_tries:
                raise ValueError(f"Attempted to solve system {num_tries} "
                                    f"times without success.")


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

    #TODO does this blow up the memory too much? is there a better way?
    inputs = [(key_hashes, values, codedict) for key_hashes, values in hash_store.buckets()]

    with Pool() as pool:
        # a list, (solution, seed) for each CSF, one per bucket
        solutions_and_seeds = pool.starmap(construct_csf_for_bucket, inputs)

    return CSF(vectorizer, hash_store.seed, solutions_and_seeds, symbols, code_length_counts)


def empirical_entropy(x):
	unique_values, unique_counts = np.unique(x, return_counts=True)
	num_entries = np.sum(unique_counts)

	sorted_indices = unique_counts.argsort()
	sorted_values = unique_values[sorted_indices[::-1]]
	sorted_counts = unique_counts[sorted_indices[::-1]]
	sorted_probs = sorted_counts / num_entries

	return -1 * np.sum(sorted_probs * np.log2(sorted_probs))


if __name__ == '__main__':
    f = open("1.4M_amazon_values.npy", "rb")
    values = np.load(f)
    keys = []
    for i in range(values.shape[0]):
        keys.append(str(i).ljust(len(str(values.shape[0])), "-"))

    t0 = time.time()
    csf = construct_csf(keys, values)
    t1 = time.time()

    print(f"Construction complete. Elapsed {t1 - t0:.2f} seconds. ")
    print(f"CSF Size is {csf.size()} bytes")

    t0 = time.perf_counter()
    csf.query(keys[0])
    t1 = time.perf_counter()

    print(f"Query time is {t1 - t0} seconds. ")

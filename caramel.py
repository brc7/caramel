from prototype.BucketedHashStore import BucketedHashStore
from prototype.Codec import make_canonical_huffman
from prototype.Modulo2System import DenseModulo2System, SparseModulo2System, UnsolvableSystemException
import math
import spookyhash

def construct_modulo2_system(key_hashes, values, codedict, seed, verbose=False):
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

    num_equations = sum(len(bitarray) for bitarray in codedict.values())
    num_variables = math.ceil(num_equations * DELTA)

    sparse_system = SparseModulo2System(num_variables)
    
    equation_id = 0
    for i, key_hash in enumerate(key_hashes):

        start_var_locations = []
        for _ in range(3):
            #TODO lets do sebastiano's trick here instead of modding
            #TODO lets write a custom hash function that generates three 64 bit 
            # hashes instead of hashing 3 times
            start_var_locations.append(
                spookyhash.hash64(int.to_bytes(key_hash, 64, "big"), seed) % num_variables
            )
            seed += 1
        
        if verbose:
            print(f"Constructing Equations for value: {values[i]} with code {codedict[values[i]]}")

        for offset, bit in enumerate(codedict[values[i]]):
            participating_variables = []
            for start_var_location in start_var_locations:
                var_location = start_var_location + offset
                if var_location >= num_variables:
                    var_location -= num_variables
                participating_variables.append(var_location)
            
            if verbose:
                print(f"    id: {equation_id}, vars: {participating_variables}, bit: {bit}")

            sparse_system.addEquation(equation_id, participating_variables, bit)
            equation_id += 1
            
    return sparse_system



def construct_csf(keys, values):
    """
    Constructs a compressed static function. This implementation sacrifices a 
    lot of the configurability for the sake of simplicity.

	Arguments:
		keys: An iterable collection of N unique hashable items.
		values: An iterable collection of N values. 

	Returns:
		A csf structure supporting the .query(key) method.
    """

    vectorizer = lambda s : bytes(s, 'utf-8')
    hash_store = BucketedHashStore(vectorizer, keys, values)
    buckets = hash_store.buckets()
    for key_hashes, values in buckets:
        codedict = make_canonical_huffman(values, verbose=True)
    
        seed = 0
        max_num_attempts = 0
        num_tries = 10
        while True:
            try: 
                sparse_system = construct_modulo2_system(key_hashes, values, codedict, seed, verbose=True)



                # peel_hypergraph(sparse_system)

                # lazy_gaussian_elimination()

                # gaussian_elimination()

                # back_substitution()

                break
            except UnsolvableSystemException as e:
                # system construction increments the seed and hashes 3 times, 
                # we add 3 here to ensure hashes are unique across runs
                seed += 3
                max_num_attempts += 1

                if max_num_attempts == num_tries:
                    raise ValueError(f"Attempted to solve system {num_tries} "
                                     f"times without success.")

    # return csf # class with .query() method. 


if __name__ == '__main__':
    keys = ["key_1", "key_2", "key_3", "key_4", "key_5"]
    values = [111, 222, 333, 444, 555]

    csf = construct_csf(keys, values)

    # for key, value in zip(keys, values):
    #     assert csf.query(key) == value
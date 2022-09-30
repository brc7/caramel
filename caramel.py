import numpy as np

def construct_modulo2_system(keys, encoded_values):
    """
    Constructs a binary system of linear equations to solve for each bit of the 
    encoded values for each key. 

	Arguments:
		keys: An iterable collection of N unique hashable items.
		encoded_values: An iterable collection of N encoded values. 

	Returns:
		A tuple of equation, value, ...???
    """

    # the resulting system has a number of equations equal to:
    #  - sum(len(encoded_value) for encoded_value in encoded_values)
    # and a length of each equation equal to:
    #  - the number of equations * SOME_CONSTANT_I_FORGOT

    system = Modulo2System() # TODO: how should we design this datastructure???
    
    for i, key in enumerate(keys):
        # hash key with 3 different hash functions, modded to the length of each equation
        # create an equation with all 0s except in the 3 locations specified by the hash, 
        # where there will be 1s

        for hash_offset, bit in enumerate(encoded_values[i]):
            # add hash_offset to each of the 3 hash locations. get 3 new hash locations

            # create an equation where each of these new hash locations is a 1

            # set the value of this equation to bit

            # add this equation to system
            pass
    
    return system


#TODO: can we optimize this function?
def scalarProduct(array1, array2):
    # returns the bitwise and of two numpy arrays (not just the integer values 
    # but the number of 1s that overlap in both array1 and array2)
    result = np.bitwise_and(array1, array2)
    sum = 0
    for number in result:
        if number != 0:
            sum += bin(number).count("1")
    return sum


def regular_gaussian_elimination(dense_system, relevant_equation_ids):
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


def construct_csf(keys, values):
    """
    Constructs a compressed static function. This implementation sacrifices a lot of the 
	configurability for the sake of simplicity.

	Arguments:
		keys: An iterable collection of N unique hashable items.
		values: An iterable collection of N values. 

	Returns:
		A csf structure supporting the .query(key) method.
    """

    encoded_values = encode_values(values)

    # Eventually we would divide keys into buckets to partition the input into a
    # subset of smaller csfs (partition depending on the hash value of the keys). 
    # This reduces the problem from one big linear system to N small linear 
    # systems. Since the complexity is O(N^3) this should significantly speed up
    # the algorithm. For now though lets just start with one linear system.


    system = construct_modulo2_system(keys, encoded_values)

    # TODO: how do we pass data between each of these steps?

    hypergraph_peeling() # how??? 

    lazy_gaussian_elimination()

    regular_gaussian_elimination(dense_system, relevant_equation_ids)

    back_substitution() # how?? what datastructures to keep track of everything?



    return csf # class with .query() method. 




if __name__ == '__main__':
    keys = ["key_1", "key_2", "key_3", "key_4", "key_5"]
    values = [111, 222, 333, 444, 555]

    csf = construct_csf(keys, values)

    for key, value in zip(keys, values):
        assert csf.query(key) == value
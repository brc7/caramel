from prototype.BucketedHashStore import BucketedHashStore
from prototype.Codec import make_canonical_huffman
from prototype.HypergraphPeeler import peel_hypergraph
from prototype.LazyGaussianElimination import lazy_gaussian_elimination
from prototype.GaussianElimination import gaussian_elimination
from prototype.Modulo2System import DenseModulo2System, SparseModulo2System

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

    system = SparseModulo2System()
    
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
        codedict = make_canonical_huffman(values)
    
        sparse_system = construct_modulo2_system(keys, [codedict[key] for key in keys])

        # peel_hypergraph(sparse_system)

        # lazy_gaussian_elimination()

        # gaussian_elimination()

        # back_substitution()



    return csf # class with .query() method. 




if __name__ == '__main__':
    keys = ["key_1", "key_2", "key_3", "key_4", "key_5"]
    values = [111, 222, 333, 444, 555]

    csf = construct_csf(keys, values)

    for key, value in zip(keys, values):
        assert csf.query(key) == value
import spookyhash

from prototype.Modulo2System import DenseModulo2System

class CSF:
    def __init__(self,
                 bucketed_hash_store_vectorizer,
                 bucketed_hash_store_seed,
                 solutions,
                 solution_sizes,
                 construction_seeds,
                 symbols,
                 bit_length_frequencies):

        self._bucketed_hash_store_vectorizer = bucketed_hash_store_vectorizer
        self._bucketed_hash_store_seed = bucketed_hash_store_seed

        # one solution per bucket
        self._solutions = solutions
        self._solution_sizes = solution_sizes
        self._construction_seeds = construction_seeds

        self._symbols = symbols
        self._bit_length_frequencies = bit_length_frequencies

    def query(self, key):
        #TODO this function repeats a lot of code. let's refactor this 
        # eventually to not duplicate code. Also let's optimize this function

        # this is copied from bucketed hash store, lets reuse
        signature = spookyhash.hash128(self._bucketed_hash_store_vectorizer, self._bucketed_hash_store_seed)
        # Use first 64 bits of the signature to identify the segment
        bucket_hash = signature >> 64
        num_buckets = len(self._solutions)
        # This outputs a uniform integer from [0, num_buckets]
        bucket_id = (bucket_hash >> 1) * (num_buckets << 1)
        bucket_id = bucket_id >> 64

        solution = self._solutions[bucket_id]
        solution_size = self._solution_sizes[bucket_id]
        construction_seed = self._construction_seeds[bucket_id]

        # this is copied from construct modulo 2 system, lets reuse
        active_var_locations = []
        for _ in range(3):
            active_var_locations.append(
                spookyhash.hash64(int.to_bytes(signature, 64, "big"), construction_seed) % solution_size
            )
            construction_seed += 1


        

        

        # XOR with the solution to get the constant
        # keep incrementing the active bit indices until you get a conclusive symbol from it


import spookyhash
from caramel.Codec import canonical_decode
from caramel.BucketedHashStore import get_bucket_id


class CSF:
    def __init__(self,
                 bucketed_hash_store_vectorizer,
                 bucketed_hash_store_seed,
                 solutions_and_seeds,
                 symbols,
                 code_length_counts):

        self._bucketed_hash_store_vectorizer = bucketed_hash_store_vectorizer
        self._bucketed_hash_store_seed = bucketed_hash_store_seed

        # a list, (solution, seed) for each CSF, one per bucket
        self._solutions_and_seeds = solutions_and_seeds

        self._symbols = symbols
        self._code_length_counts = code_length_counts

    def size(self) -> int:
        size = 0
        import sys 
        size += sys.getsizeof(self._bucketed_hash_store_vectorizer)
        size += sys.getsizeof(self._bucketed_hash_store_seed)
        size += sys.getsizeof(self._solutions_and_seeds)
        size += sys.getsizeof(self._symbols)
        size += sys.getsizeof(self._code_length_counts)
        return size

    def query(self, key):
        signature = spookyhash.hash128(self._bucketed_hash_store_vectorizer(key), self._bucketed_hash_store_seed)
        bucket_id = get_bucket_id(signature, num_buckets=len(self._solutions_and_seeds))

        solution, construction_seed = self._solutions_and_seeds[bucket_id]
        solution_size = len(solution)

        # The general idea is to hash the signature 3 times to get 3 initial
        # locations in the solution bitarray. Then for each location, we will 
        # read a chunk of the solution bitarray the size of max_codelength. We 
        # will then XOR these three chunks to get some encoded value. We will 
        # then loop through this value and decode it with the canonical decoder.  

        max_codelength = len(self._code_length_counts) - 1
        sections_to_xor = []
        temp_seed = construction_seed
        for _ in range(3):
            location = spookyhash.hash64(int.to_bytes(signature, 64, "big"), temp_seed) % solution_size
            temp_seed += 1

            # wrap around in case the location we hashed to is too big
            # TODO: can we optimize this wrap around?
            if location + max_codelength >= solution_size:
                sections_to_xor.append(solution[location:] + solution[:max_codelength - (solution_size - location) + 1])
            else:
                sections_to_xor.append(solution[location:location + max_codelength + 1])

        section1, section2, section3 = sections_to_xor
        encoded_value = section1 ^ section2 ^ section3

        return canonical_decode(encoded_value, self._code_length_counts, self._symbols)

        


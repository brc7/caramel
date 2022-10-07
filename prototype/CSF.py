import spookyhash
import numpy as np

class CSF:
    def __init__(self,
                 bucketed_hash_store_vectorizer,
                 bucketed_hash_store_seed,
                 solutions,
                 solution_sizes,
                 construction_seeds,
                 symbols,
                 code_length_counts):

        self._bucketed_hash_store_vectorizer = bucketed_hash_store_vectorizer
        self._bucketed_hash_store_seed = bucketed_hash_store_seed

        # one solution per bucket
        self._solutions = solutions
        self._solution_sizes = solution_sizes
        self._construction_seeds = construction_seeds

        self._symbols = symbols
        self.code_length_counts = code_length_counts

    def query(self, key):
        #TODO this function repeats a lot of code. let's refactor this 
        # eventually to not duplicate code. Also let's optimize this function

        # this is copied from bucketed hash store, lets reuse
        signature = spookyhash.hash128(self._bucketed_hash_store_vectorizer(key), self._bucketed_hash_store_seed)
        # Use first 64 bits of the signature to identify the segment
        bucket_hash = signature >> 64
        num_buckets = len(self._solutions)
        # This outputs a uniform integer from [0, num_buckets]
        bucket_id = (bucket_hash >> 1) * (num_buckets << 1)
        bucket_id = bucket_id >> 64

        solution = self._solutions[bucket_id]
        solution_size = self._solution_sizes[bucket_id]
        construction_seed = self._construction_seeds[bucket_id]

        # The general idea is to hash the signature 3 times to get 3 initial
        # locations in the solution bitarray. Then for each location, we will 
        # read a chunk of the solution bitarray the size of max_codelength. We 
        # will then XOR these three chunks to get some bitarray. We will then 
        # loop through this bitarray and report the first valid value from
        # from decoding it with the canonical huffman decoder.  

        max_codelength = len(self.code_length_counts) - 1
        sections_to_xor = []
        for _ in range(3):
            location = spookyhash.hash64(int.to_bytes(signature, 64, "big"), construction_seed) % solution_size
            construction_seed += 1

            # we need to wrap around the solution
            if location + max_codelength >= solution_size:
                print("WRAP")
                print(solution[location:])
                print(solution[:max_codelength - (solution_size - location) + 1])
                sections_to_xor.append(solution[location:] + solution[:max_codelength - (solution_size - location) + 1])
            else:
                sections_to_xor.append(solution[location:location + max_codelength + 1])
            
            print(solution)
            print(location)
            print(solution_size)
            print(sections_to_xor)
            
        section1, section2, section3 = sections_to_xor
        encoded_value = section1 ^ section2 ^ section3

        # decode the encoded_value
        code = 0
        first = 0
        index = 0
        for i in range(1, max_codelength + 1):
            next_bit = encoded_value[i - 1]
            code = np.bitwise_or(code, next_bit)

            # number of codes of bitlength i
            count = self.code_length_counts[i]

            if code - count < first:
                return self._symbols[index + (code - first)]
            
            index += count
            first += count
            first <<= 1
            code <<= 1

        return "ERROR HERE"

        


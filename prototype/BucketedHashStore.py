import spookyhash

# For usage example
import random
import string
import math

class BucketedHashStore:
    def __init__(self,
            vectorizer,
            keys,
            values,
            bucket_size: int = 256,
            num_attempts: int = 3,
        ):
        """ Construct a sharded store of key signatures and values, where
        the pairs are grouped into buckets of a given size (in expectation).
        Keys are sorted into buckets based on the first 64 bits of their hash.

        Arguments:
            vectorizer: A callable that transforms keys into a bytes object
                that can be passed to spookyhash.hash128().
            keys: An iterable of N unique keys.
            values: An iterable of N values (arbitrary, not just integers).
            bucket_size: Integer. An upper bound on the expected number of
                items in each bucket.
            num_attempts: Integer. The number of times to retry construction.
                The construction fails if any keys collide under the hash,
                which happens by chance with small probability or if there are
                duplicate keys. If num_attempts fail, then it is highly likely
                that there are duplicate keys.

        Raises: 
            ValueError if we make num_attempts to construct the hash and all
            attempts fail.
        """
        self._vectorizer = vectorizer
        self._num_keys = len(keys)
        self._bucket_size = bucket_size
        self._num_buckets = 1 + self._num_keys // self._bucket_size
        self._seed = 0  # Will be incremented until collisions do not occur.
        self._key_buckets = tuple([[] for _ in range(self._num_buckets)])
        self._value_buckets = tuple([[] for _ in range(self._num_buckets)])
        
        is_constructed = False
        while not is_constructed:
            try:
                self._construct(keys, values)
                is_constructed = True
            except ValueError as e:
                if self._seed < num_attempts:
                    self._seed += 1
                    self._key_buckets = tuple([[] for _ in range(self._num_buckets)])
                    self._value_buckets = tuple([[] for _ in range(self._num_buckets)])
                else:
                    raise e

    def _construct(self, keys, values):
        for key, value in zip(keys, values):
            self._add(key, value)

    def _add(self, key, value):
        signature = spookyhash.hash128(self._vectorizer(key), self._seed)
        bucket_id = get_bucket_id(signature, self._num_buckets)
        if signature in self._key_buckets[bucket_id]:
            raise ValueError("Detected a key collision under 128-bit hash. "
                             "Likely due to a duplicate key.")
        self._key_buckets[bucket_id].append(signature)
        self._value_buckets[bucket_id].append(value)

    def buckets(self):
        """ Returns an iterator over pairs of key signatures and values.

        Returns:
            An iterator over  (signatures, values) pairs, where
            signatures is a list of N key signatures and values is a list of
            the N corresponding values.
        """
        return zip(self._key_buckets, self._value_buckets)

    @property
    def seed(self):
        return self._seed


def get_bucket_id(signature, num_buckets):
    # Use first 64 bits of the signature to identify the segment
    bucket_hash = signature >> 64
    # This outputs a uniform integer from [0, self._num_buckets]
    bucket_id = (bucket_hash >> 1) * (num_buckets << 1)
    bucket_id = bucket_id >> 64
    return bucket_id


if __name__ == "__main__":
    num_keys = 1000
    keys = [str(i) for i in range(num_keys)]
    values = [math.floor(math.log(random.randint(1, 100)))
              for _ in range(num_keys)]

    vectorizer = lambda s : bytes(s, 'utf-8')
    hash_store = BucketedHashStore(vectorizer, keys, values)
    buckets = hash_store.buckets()
    for keys, values in buckets:
        print(len(keys))

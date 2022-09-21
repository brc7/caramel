



class Huffman:
    def __init__(self,
            symbols,
            frequencies,
            max_table_length = None,
            entropy_threshold = None,
        ):
        self._max_table_length = max_table_length
        self._entropy_threshold = entropy_threshold

        self._vocab_size = 


        # 1. Sort symbols (in ascending order).




        # 2. 


        # Reference implementation in C:
        # https://github.com/madler/brotli/blob/master/huff.c




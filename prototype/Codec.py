import numpy as np
from collections import defaultdict
from bitarray.util import canonical_huffman, canonical_decode, int2ba

def calculate_frequencies(symbols):
    # returns a map from symbol to frequency
    frequency_map = defaultdict(int)
    for symbol in symbols:
        frequency_map[symbol] += 1
    return frequency_map


def min_redundancy_codeword_lengths(frequencies):
    """
    frequencies: list of symbol frequencies sorted in non-decreasing order, 
        first by frequency, then by symbol. Its important to sort by symbol as
        a tiebreaker so the decoder can reconstruct the codes
    returns: the expected lengths of codewords in huffman encoding

    Algorithm described in: http://hjemmesider.diku.dk/~jyrki/Paper/WADS95.pdf
    reference sources: 
     - https://github.com/madler/brotli/blob/master/huff.c
     - https://people.eng.unimelb.edu.au/ammoffat/inplace.c
    """

    # uses name A just to follow the algorithm
    A = list(frequencies) # TODO do we need to copy here??
    size = len(A)

    # check trivial cases
    if size == 0:
        return A
    if size == 1:
        A[0] = 1
        return A
     
    # first pass, left to right, setting parent pointers
    A[0] += A[1] # TODO sebastiano does A[0] = 1???
    root = 0
    leaf = 2
    for next in range(1, size - 1):
        # select first item for a pairing
        if leaf >= size or A[root] < A[leaf]:
            A[next] = A[root]
            A[root] = next
            root += 1
        else:
            A[next] = A[leaf]
            leaf += 1

        # add on the second item
        if leaf >= size or (root < next and A[root] < A[leaf]):
            A[next] += A[root]
            A[root] = next
            root += 1
        else:
            A[next] += A[leaf]
            leaf += 1
    
    # second pass, right to left, setting internal depths
    A[size - 2] = 0
    for next in range(size - 3, -1, -1):
        A[next] = A[A[next]] + 1
    
    # third pass, right to left, setting internal depths
    available = 1
    used = 0
    depth = 0
    root = size - 2
    next = size - 1
    while available > 0:
        while root >= 0 and A[root] == depth:
            used += 1
            root -= 1
        while available > used:
            A[next] = depth
            next -= 1
            available -= 1
        available = 2 * used
        depth += 1
        used = 0
    
    return A


def make_canonical_huffman(symbols, verbose = False):
    frequency_map = calculate_frequencies(symbols)
    # we sort in non-decreasing order, first by frequency then by symbol. 
    # this is required for the decoder to reconstruct the codes
    symbol_frequency_pairs = sorted(frequency_map.items(), key=lambda sym_freq: (sym_freq[1], sym_freq[0]))
    codeword_lengths = min_redundancy_codeword_lengths([x[1] for x in symbol_frequency_pairs])

    # reverse because we should do code assignment in non-decreasing order of 
    # bit length instead of frequency
    symbol_frequency_pairs.reverse()
    codeword_lengths.reverse()    

    # TODO add length limiting?
    codedict = {}
    code = 0
    current_length = codeword_lengths[0]
    for i, (symbol, frequency) in enumerate(symbol_frequency_pairs):
        current_length = codeword_lengths[i]
        codedict[symbol] = int2ba(code, length=current_length, endian='big')
        if i + 1 < len(codeword_lengths):
            code += 1
            code <<= codeword_lengths[i + 1] - current_length

    if verbose:
        print(f"Canonical huffman produced codedict: {codedict}")

    return codedict #TODO we'll need to return other things for the decoder


def test_canonical_huffman(values):
    # our implementation
    actual_codedict = make_canonical_huffman(values)

    # bitarray's implementation
    frequency_map = calculate_frequencies(values)
    expected_codedict, counts, symbols = canonical_huffman(frequency_map)

    # can't do == on the maps because two identical bitarrays aren't equal
    for actual_key, expected_key in zip(sorted(actual_codedict.keys()), sorted(expected_codedict.keys())):
        assert actual_key == expected_key
        assert actual_codedict[actual_key].to01(), expected_codedict[expected_key].to01()


if __name__ == "__main__":
    for i in range(100):
        symbols = np.random.randint(0, 20, size=30)    
        test_canonical_huffman(symbols)
from collections import defaultdict


def calculate_frequencies(values):
    # returns a list of value frequencies in non-decreasing order
    frequency_map = defaultdict(int)
    for value in values:
        frequency_map[value] += 1
    return sorted(frequency_map.values())


def min_redundancy_codeword_lengths(frequencies):
    """
    frequencies: list of symbol frequencies in non-decreasing order
    returns: the expected minimum redundancy codes

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


if __name__ == "__main__":
    values = [0, 0, 0, 0, 0,
              1, 1, 1, 1,
              2, 2, 2,
              3, 3, 
              4, 
              5]

    frequencies = calculate_frequencies(values)
    codes = min_redundancy_codeword_lengths(frequencies)
    print(f"Frequencies: {frequencies}, Codeword lengths: {codes}")

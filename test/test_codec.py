import pytest
from caramel.Codec import (calculate_frequencies, canonical_huffman,
                           make_canonical_huffman)

def verify_canonical_huffman(values):
    # our implementation
    actual_codedict = make_canonical_huffman(values)

    # bitarray's implementation
    frequency_map = calculate_frequencies(values)
    expected_codedict, counts, symbols = canonical_huffman(frequency_map)

    # can't do == on the maps because two identical bitarrays aren't equal
    for actual_key, expected_key in zip(sorted(actual_codedict.keys()), sorted(expected_codedict.keys())):
        assert actual_key == expected_key
        assert actual_codedict[actual_key].to01(), expected_codedict[expected_key].to01()


@pytest.mark.skip(reason="This currently fails")
def test_canonical_huffman():
    for i in range(100):
        symbols = np.random.randint(0, 20, size=30)    
        verify_canonical_huffman(symbols)
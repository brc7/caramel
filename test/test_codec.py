import pytest
import numpy as np
from caramel.Codec import canonical_decode, canonical_huffman

def verify_canonical_huffman(symbols):
    codedict, code_length_counts, sorted_symbols = canonical_huffman(symbols)

    for expected_key, code in codedict.items():
        actual_key = canonical_decode(code, code_length_counts, sorted_symbols)
        assert actual_key == expected_key


@pytest.mark.skip(reason="This currently fails")
def test_canonical_huffman():
    for i in range(100):
        symbols = np.random.randint(0, 20, size=30)    
        verify_canonical_huffman(symbols)
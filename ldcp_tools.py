from config import LDPC_CODER

import numpy as np


def encode_bytes(data: bytes):
    data = np.array(list(data), dtype=np.uint8)
    binary_data = np.unpackbits(data)

    padding_needed = (LDPC_CODER.K - binary_data.size % LDPC_CODER.K) % LDPC_CODER.K

    binary_data = np.concatenate([binary_data, [0] * padding_needed])

    all_encoded_binary_data = []
    chunks_to_encode = np.split(binary_data, binary_data.size // LDPC_CODER.K)
    for chunk in chunks_to_encode:
        all_encoded_binary_data.extend(LDPC_CODER.encode(chunk))

    encoded_data = np.packbits(all_encoded_binary_data)

    return bytes(encoded_data)


def decode_from_llr(llr: np.ndarray):
    excess_llr_blocks = llr.size % LDPC_CODER.N
    cropped_llr = llr[:-excess_llr_blocks]

    llr_blocks_for_jossy = np.split(cropped_llr, cropped_llr.size // LDPC_CODER.N)

    binary_data = []
    for block in llr_blocks_for_jossy:
        app, _ = LDPC_CODER.decode(block)
        binary_data.extend((app < 0)[:LDPC_CODER.K])

    return bytes(np.packbits(np.array(binary_data, dtype=np.uint8)))

import numpy as np
import sys


def csv_column_to_numpy(filename):
    return np.fromfile(filename, sep="\n")


def get_gray_code_from_symbol(complex_noisy_symbol):
    if complex_noisy_symbol.real > 0:
        if complex_noisy_symbol.imag > 0:
            return 0
        else:
            return 2
    else:  # real < 0
        if complex_noisy_symbol.imag > 0:
            return 1
        else:
            return 3


def get_bytes_from_noisy_symbols(symbols):
    output_num = 0
    for index, symbol in enumerate(symbols):
        index_in_local_bin = index % 1024
        if index_in_local_bin > 511 or index_in_local_bin == 0:
            continue

        output_num <<= 2
        output_num |= get_gray_code_from_symbol(symbol)

    return output_num.to_bytes((len(symbols) - 2) // 8, byteorder="big")


N = 1024
L = 32

channel_response = csv_column_to_numpy("channel_impulse.csv")
received_signal = csv_column_to_numpy(sys.argv[1])

dft_of_channel_response = np.fft.fft(channel_response, N)
blocks_of_received_signal = np.split(received_signal, len(received_signal) // (N + L))

for i in range((4 - len(blocks_of_received_signal) % 4) % 4):
    blocks_of_received_signal.append(np.zeros(N + L))

bytes_of_file = bytes()

for block_index in range(0, len(blocks_of_received_signal), 4):
    # nibble bad, byte good: ensure the file is a multiple of 8 bits
    dft_blocks = []
    for i in range(4):
        block_without_cyclic_prefix = blocks_of_received_signal[block_index + i][32:]
        dft_blocks.append(
            np.fft.fft(block_without_cyclic_prefix, N) / dft_of_channel_response
        )

    bytes_of_file += get_bytes_from_noisy_symbols(np.concatenate(dft_blocks))


open(f"{sys.argv[1].strip('.csv')}.bin", "wb").write(bytes_of_file)

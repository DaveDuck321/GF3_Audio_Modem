from config import (
    CONSTELLATION_SYMBOLS,
    CONSTELLATION_BITS,
    OFDM_BODY_LENGTH,
    OFDM_CYCLIC_PREFIX_LENGTH,
)
from signal_builder import SignalBuilder

import numpy as np

import random

def map_to_constellation_symbols(data: bytes):
    constellation_symbols = []

    current_group = 0
    current_group_size = 0
    for byte in data:
        for shift in range(8, 0, -1):
            current_group <<= 1
            current_group_size += 1

            current_group |= 1 & (byte >> (shift - 1))

            if current_group_size == CONSTELLATION_BITS:
                constellation_symbols.append(CONSTELLATION_SYMBOLS[current_group])

                current_group_size = 0
                current_group = 0

    # TODO: we can pad with zeros here
    assert current_group == 0
    assert current_group_size == 0

    return constellation_symbols


def pad_symbols(symbols, alignment):
    unmatched_symbols = len(symbols) % alignment

    padding_count = (alignment - unmatched_symbols) % alignment

    for i in range(padding_count):
        symbols.append(random.choice(CONSTELLATION_SYMBOLS))

    return symbols


def modulate_bytes(data: bytes):
    ofdm_data_length = OFDM_BODY_LENGTH // 2 - 1

    constellation_symbols = map_to_constellation_symbols(data)
    constellation_symbols = pad_symbols(constellation_symbols, ofdm_data_length)

    data_blocks = np.split(
        np.array(constellation_symbols),
        len(constellation_symbols) // (OFDM_BODY_LENGTH // 2 - 1),
    )

    ofdm_signal = SignalBuilder()
    for block in data_blocks:
        # Information is only mapped to frequency bins 1 to 511.
        # Frequency bins 513 to 1023 contain the reverse ordered conjugate
        # complex values of frequency bins 1 to 511. Frequency bins 0 and 512
        # contain 0 (no information, value 0.) This all ensures that the
        # output of the OFDM modulator is a real (baseband) vector.

        block_for_real_transmission = np.concatenate(
            [[0], block, [0], np.conjugate(block[-1::-1])]
        )
        assert block_for_real_transmission.size == OFDM_BODY_LENGTH

        idft_of_block = np.fft.ifft(block_for_real_transmission, OFDM_BODY_LENGTH)
        block_with_cyclic_prefix = np.concatenate(
            [idft_of_block[-OFDM_CYCLIC_PREFIX_LENGTH:], idft_of_block]
        )

        # Ensure imaginary component is zero
        assert not block_with_cyclic_prefix.imag.any()

        ofdm_signal.append_signal_part(block_with_cyclic_prefix.real)

    return ofdm_signal.get_signal()


def map_received_constellation_symbol_to_value(symbol):
    return sorted(
        CONSTELLATION_SYMBOLS,
        key=lambda value: abs(symbol - CONSTELLATION_SYMBOLS[value]),
    )[0]


def demodulate_signal(channel_coefficients, signal):
    ofdm_blocks = np.split(
        signal, signal.size // (OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH)
    )  # TODO: WHAT HAPPENS IF WE LOSE A SAMPLE HERE?

    channel_dft = np.fft.fft(channel_coefficients, OFDM_BODY_LENGTH)

    curret_byte = 0
    position_in_current_byte = 0
    output_bytes = []

    for block in ofdm_blocks:
        block_without_cyclic_prefix = block[OFDM_CYCLIC_PREFIX_LENGTH:]
        dft_of_block = np.fft.fft(block_without_cyclic_prefix, OFDM_BODY_LENGTH)

        equalized_dft = dft_of_block / channel_dft

        for index, noisy_symbol in enumerate(equalized_dft):
            if index == 0:
                continue  # The first element is always set to zero: it contains no data

            if index == OFDM_BODY_LENGTH // 2:
                break  # TODO: use conjugate pairs for error correction?

            position_in_current_byte += CONSTELLATION_BITS
            curret_byte <<= CONSTELLATION_BITS
            curret_byte |= map_received_constellation_symbol_to_value(noisy_symbol)
            
            if position_in_current_byte == 8:
                output_bytes.append(curret_byte)
                curret_byte = 0
                position_in_current_byte = 0

    # assert curret_byte == 0
    # assert position_in_current_byte == 0

    return bytes(output_bytes)

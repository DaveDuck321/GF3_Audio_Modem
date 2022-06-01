from config import CHIRP, OFDM_BODY_LENGTH, OFDM_CYCLIC_PREFIX_LENGTH
import OFDM

import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt


def get_chirp_indices(transmitted_signal: np.ndarray):
    convolved = signal.convolve(transmitted_signal, CHIRP[::-1])

    peaks, _ = signal.find_peaks(convolved, distance=CHIRP.size - 1)

    highest_peaks = sorted(
        sorted(peaks, key=lambda index: convolved[index], reverse=True)[:4]
    )

    # Check the peaks are spaced about right
    assert abs(((highest_peaks[1] - highest_peaks[0]) - CHIRP.size)) < 0.1 * CHIRP.size
    assert abs(((highest_peaks[3] - highest_peaks[2]) - CHIRP.size)) < 0.1 * CHIRP.size

    initial_chirps_start_index = (
        highest_peaks[0] - CHIRP.size + (highest_peaks[1] - 2 * CHIRP.size)
    ) // 2

    final_chirps_start_index = (
        highest_peaks[2] - CHIRP.size + (highest_peaks[3] - 2 * CHIRP.size)
    ) // 2

    return (initial_chirps_start_index, final_chirps_start_index)


def crop_signal_into_parts(transmitted_signal: np.ndarray):
    initial_chirps_start_index, final_chirps_start_index = get_chirp_indices(
        transmitted_signal
    )

    calculated_ofdm_length = (
        final_chirps_start_index - initial_chirps_start_index - 2 * CHIRP.size
    )
    ofdm_block_length = OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH
    number_of_ofdm_blocks = round(calculated_ofdm_length / ofdm_block_length)

    true_ofdm_length = number_of_ofdm_blocks * ofdm_block_length
    start_of_ofdm_block = initial_chirps_start_index + 2 * CHIRP.size

    return (
        transmitted_signal[
            max(initial_chirps_start_index, 0) : initial_chirps_start_index
            + 2 * CHIRP.size
        ],
        transmitted_signal[
            start_of_ofdm_block : start_of_ofdm_block + 4*ofdm_block_length
        ],
        transmitted_signal[
            start_of_ofdm_block + 4*ofdm_block_length : start_of_ofdm_block + true_ofdm_length
        ],
        transmitted_signal[
            final_chirps_start_index : final_chirps_start_index + 2 * CHIRP.size
        ],
    )


def estimate_channel_coefficients(recorded_known_ofdm_blocks: np.ndarray):
    print(len(recorded_known_ofdm_blocks))
    known_ofdm_block = OFDM.generate_known_ofdm_block()
    fft_of_known_block = np.fft.fft(known_ofdm_block[OFDM_CYCLIC_PREFIX_LENGTH:], OFDM_BODY_LENGTH)

    sum_of_gains = np.zeros(OFDM_BODY_LENGTH, dtype=np.complex128)
    known_blocks = np.split(recorded_known_ofdm_blocks, 4)
    for block in known_blocks:
        sum_of_gains += estimate_frequency_gains_from_block(block, fft_of_known_block)

    return np.fft.ifft(sum_of_gains / 4, OFDM_BODY_LENGTH)


def estimate_frequency_gains_from_block(recorded_block: np.ndarray,  expected_block_fft: np.ndarray):
    fft_of_recorded_block = np.fft.fft(recorded_block[OFDM_CYCLIC_PREFIX_LENGTH:], OFDM_BODY_LENGTH)
    return fft_of_recorded_block / expected_block_fft

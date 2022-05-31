from config import CHIRP, OFDM_BODY_LENGTH, OFDM_CYCLIC_PREFIX_LENGTH
import OFDM

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d

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
            start_of_ofdm_block : start_of_ofdm_block + ofdm_block_length
        ],
        transmitted_signal[
            start_of_ofdm_block + ofdm_block_length : start_of_ofdm_block + true_ofdm_length
        ],
        transmitted_signal[
            final_chirps_start_index : final_chirps_start_index + 2 * CHIRP.size
        ],
    )


def estimate_channel_coefficients(recorded_known_ofdm_block: np.ndarray):
    KNOWN_OFDM_BLOCK = OFDM.generate_known_ofdm_block()[OFDM_CYCLIC_PREFIX_LENGTH:]
    fft_of_true_block = np.fft.fft(KNOWN_OFDM_BLOCK, OFDM_BODY_LENGTH)
    fft_of_recorded_block = np.fft.fft(recorded_known_ofdm_block[OFDM_CYCLIC_PREFIX_LENGTH:], OFDM_BODY_LENGTH)

    frequency_response = fft_of_recorded_block / fft_of_true_block
    return np.fft.ifft(frequency_response, OFDM_BODY_LENGTH)

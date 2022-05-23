from config import CHIRP, OFDM_BODY_LENGTH, OFDM_CYCLIC_PREFIX_LENGTH

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
            start_of_ofdm_block : start_of_ofdm_block + true_ofdm_length
        ],
        transmitted_signal[
            final_chirps_start_index : final_chirps_start_index + 2 * CHIRP.size
        ],
    )


def estimate_channel_coefficients(chirp_signal: np.ndarray):
    convolved_1 = signal.convolve(chirp_signal[: CHIRP.size], CHIRP[::-1])
    channel_coefficients_1 = convolved_1[CHIRP.size : CHIRP.size + OFDM_CYCLIC_PREFIX_LENGTH]

    convolved_2 = signal.convolve(chirp_signal[CHIRP.size :], CHIRP[::-1])
    channel_coefficients_2 = convolved_2[CHIRP.size : CHIRP.size + OFDM_CYCLIC_PREFIX_LENGTH]

    fft_channel_coefficients_1 = np.fft.fft(channel_coefficients_1)
    fft_channel_coefficients_2 = np.fft.fft(channel_coefficients_2)

    average_magnitude = 0.5 * (np.abs(fft_channel_coefficients_1) + np.abs(fft_channel_coefficients_2))
    average_phase = 0.5 * (np.angle(fft_channel_coefficients_1) + np.angle(fft_channel_coefficients_2))

    average_channel_coefficients = np.fft.ifft(average_magnitude * np.exp(1j * average_phase))
    combined_channel  =  np.fft.ifft(np.abs(fft_channel_coefficients_2) * np.exp(1j * np.angle(fft_channel_coefficients_2)))

    # plt.figure()
    # plt.plot(np.angle(np.fft.fft(combined_channel, OFDM_BODY_LENGTH)), color='g')
    # plt.plot(np.angle(np.fft.fft(channel_coefficients_1, OFDM_BODY_LENGTH)), color='y')
    # plt.plot(np.angle(np.fft.fft(channel_coefficients_2, OFDM_BODY_LENGTH)), color='purple')
    # plt.plot(np.angle(np.fft.fft(average_channel_coefficients, OFDM_BODY_LENGTH)), color='r')

   
    # plt.figure()
    # plt.plot(np.abs(np.fft.fft(combined_channel, OFDM_BODY_LENGTH)), color='g')
    # plt.plot(np.abs(np.fft.fft(channel_coefficients_1, OFDM_BODY_LENGTH)), color='y')
    # plt.plot(np.abs(np.fft.fft(channel_coefficients_2, OFDM_BODY_LENGTH)), color='purple')
    # plt.plot(np.abs(np.fft.fft(average_channel_coefficients, OFDM_BODY_LENGTH)), color='r')
    # plt.show()

    # print(channel_coefficients)
    return channel_coefficients_1

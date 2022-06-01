from config import (
    CHIRP,
    CHIRP_DURATION,
    KNOWN_OFDM_REPEAT_COUNT,
    NUMBER_OF_SYMBOLS_IN_FRAME,
    OFDM_BODY_LENGTH,
    OFDM_CYCLIC_PREFIX_LENGTH,
    OFDM_SYMBOL_LENGTH,
    SAMPLE_RATE,
)
import OFDM

import numpy as np
import scipy.signal as signal


def crop_signal_into_overlapping_frames(transmitted_signal: np.ndarray):
    chirp_convolution = signal.convolve(transmitted_signal, CHIRP[::-1])

    # This isn't great, but imo this is the best way to identify an unknown number of peaks
    peak_threshold = 0.75 * np.max(chirp_convolution)
    peaks, _ = signal.find_peaks(
        chirp_convolution, peak_threshold, distance=CHIRP.size - 1
    )
    peaks -= CHIRP.size  # Convert to signal coordinates

    # Ensure all peaks are reasonably spaced/ standard compliant
    assert len(peaks) % 2 == 0
    for peak_index in range(0, len(peaks), 2):
        assert (
            abs(peaks[peak_index + 1] - peaks[peak_index] - CHIRP.size)
            < 0.01 * CHIRP.size
        )

        if peak_index == 0:
            continue

        expected_min_length = CHIRP_DURATION * SAMPLE_RATE + OFDM_SYMBOL_LENGTH * (
            2 * KNOWN_OFDM_REPEAT_COUNT + NUMBER_OF_SYMBOLS_IN_FRAME["min"]
        )
        expected_max_length = CHIRP_DURATION * SAMPLE_RATE + OFDM_SYMBOL_LENGTH * (
            2 * KNOWN_OFDM_REPEAT_COUNT + NUMBER_OF_SYMBOLS_IN_FRAME["max"]
        )
        assert (
            abs(peaks[peak_index] - peaks[peak_index - 1]) > 0.99 * expected_min_length
        )
        assert (
            abs(peaks[peak_index] - peaks[peak_index - 1]) < 1.01 * expected_max_length
        )

    # Lets crop the signal
    frames = []
    if peaks[0] < 0:
        # Correct debug signals without applying a bias
        peaks -= peaks[0]

    for peak_index in range(0, len(peaks) - 2, 2):
        frames.append(
            transmitted_signal[peaks[peak_index] : peaks[peak_index + 3] + CHIRP.size]
        )

    return frames


def crop_frame_into_parts(frame: np.ndarray):
    initial_chirps_start_index = 0
    initial_chirps_end_index = 2 * CHIRP.size

    final_chirps_start_index = frame.size - 2 * CHIRP.size
    final_chirps_end_index = frame.size

    prefix_known_symbol_start = initial_chirps_end_index
    prefix_known_symbol_end = (
        prefix_known_symbol_start + KNOWN_OFDM_REPEAT_COUNT * OFDM_SYMBOL_LENGTH
    )

    endfix_known_symbol_end = final_chirps_start_index
    endfix_known_symbol_start = (
        endfix_known_symbol_end - KNOWN_OFDM_REPEAT_COUNT * OFDM_SYMBOL_LENGTH
    )

    samples_of_data_ofdm = endfix_known_symbol_start - prefix_known_symbol_end

    print(
        f"[INFO] Expected: {samples_of_data_ofdm / OFDM_SYMBOL_LENGTH} data OFDM symbols"
    )
    number_of_ofdm_blocks = round(samples_of_data_ofdm / OFDM_SYMBOL_LENGTH)

    return (
        frame[initial_chirps_start_index:initial_chirps_end_index],  # Chirp
        frame[prefix_known_symbol_start:prefix_known_symbol_end],  # Prefix
        frame[prefix_known_symbol_end:endfix_known_symbol_start],  # Data
        frame[endfix_known_symbol_start:endfix_known_symbol_end],  # Endfix
        frame[final_chirps_start_index:final_chirps_end_index],  # Chirp
    )


def estimate_channel_coefficients(recorded_known_ofdm_blocks: np.ndarray):
    known_ofdm_block = OFDM.generate_known_ofdm_block()[OFDM_CYCLIC_PREFIX_LENGTH:]
    fft_of_known_block = np.fft.fft(known_ofdm_block, OFDM_BODY_LENGTH)

    sum_of_gains = np.zeros(OFDM_BODY_LENGTH, dtype=np.complex128)
    known_blocks = np.split(recorded_known_ofdm_blocks, KNOWN_OFDM_REPEAT_COUNT)
    for block in known_blocks:
        sum_of_gains += estimate_frequency_gains_from_block(block, fft_of_known_block)

    return np.fft.ifft(sum_of_gains / KNOWN_OFDM_REPEAT_COUNT, OFDM_BODY_LENGTH)


def estimate_frequency_gains_from_block(
    recorded_block: np.ndarray, expected_block_fft: np.ndarray
):
    fft_of_recorded_block = np.fft.fft(
        recorded_block[OFDM_CYCLIC_PREFIX_LENGTH:], OFDM_BODY_LENGTH
    )
    return fft_of_recorded_block / expected_block_fft

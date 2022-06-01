# vim: set ts=4 sw=4 tw=0 et :
from config import (
    CONSTELLATION_BITS,
    CONSTELLATION_SYMBOLS,
    OFDM_BODY_LENGTH,
    OFDM_CYCLIC_PREFIX_LENGTH,
    OFDM_DATA_INDEX_RANGE,
    PEAK_SUPPRESSION_STATS_ENABLED,
    PEAK_SUPPRESSION_ENABLED,
    PEAK_SUPPRESSION_IMPULSE_APPROXIMATOR,
    PEAK_SUPPRESSION_SEQUENCE,
    SONG_LEN,
    SONG_NOTES,
    SONG_VOLUME,
    SONG,
    get_index_of_frequency,
)
from signal_builder import SignalBuilder

import numpy as np
import scipy

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

    for _ in range(padding_count):
        symbols.append(random.choice(CONSTELLATION_SYMBOLS))

    return symbols


def generate_known_ofdm_block():
    return np.load('known_ofdm_symbol.npy')

def suppress_peaks(data: np.ndarray):
    if not PEAK_SUPPRESSION_ENABLED:
        return data

    assert data.size == OFDM_BODY_LENGTH

    # this is overwritten on each pass
    improved_time_domain_block = np.copy(data)

    for thresh_std_devs, SAMPLE_VIEW_RANGE, IMPULSE_SHIFT_RANGE in PEAK_SUPPRESSION_SEQUENCE:

        peak_thresh = thresh_std_devs * np.sqrt(improved_time_domain_block.var())

        window_chunks = [] # chunks of the original signal with the peaks in them
        for sample_idx, sample in enumerate(improved_time_domain_block):
            if abs(sample) > peak_thresh:
                # expand a window if this sample is in one, else make a new window
                for i, (peak, c_min, c_max) in enumerate(window_chunks):
                    if c_min <= sample_idx + SAMPLE_VIEW_RANGE \
                            and sample_idx - SAMPLE_VIEW_RANGE <= c_max:
                        window_chunks[i] = (
                                max(abs(sample), peak),
                                min(c_min, sample_idx - SAMPLE_VIEW_RANGE),
                                max(c_max, sample_idx + SAMPLE_VIEW_RANGE)
                            )
                        break
                else:
                    window_chunks.append((
                            abs(sample),
                            sample_idx - SAMPLE_VIEW_RANGE,
                            sample_idx + SAMPLE_VIEW_RANGE
                        ))

        if len(window_chunks) == 0:
            # can't believe this edge case actually happened
            # (no peaks above the threshold)
            continue

        # combine edge windows (note the time domain is cyclic)
        first_window_peak, first_window_left, first_window_right = window_chunks[0]
        last_window_peak, last_window_left, last_window_right = window_chunks[-1]

        if last_window_right - OFDM_BODY_LENGTH >= first_window_left:
            # if they overlap, combine them on the left
            window_chunks[0] = (
                    max(first_window_peak, last_window_peak),
                    last_window_left - OFDM_BODY_LENGTH,
                    first_window_right
                )
            window_chunks.pop(-1)

        # suppress the biggest peaks first
        window_chunks.sort(reverse=True)

        for _, c_min, c_max in window_chunks:
            shifted_impulses = []
            for s in range(
                        c_min + (SAMPLE_VIEW_RANGE - IMPULSE_SHIFT_RANGE),
                        c_max - (SAMPLE_VIEW_RANGE - IMPULSE_SHIFT_RANGE)
                    ):
                shifted_impulses.append(np.roll(PEAK_SUPPRESSION_IMPULSE_APPROXIMATOR, s - OFDM_BODY_LENGTH//2))

            # deal with the edge peak case
            if c_min < 0:
                cut_time_domain_target = np.concatenate([
                        -improved_time_domain_block[c_min + OFDM_BODY_LENGTH:],
                        -improved_time_domain_block[:c_max],
                    ])
                cut_shifted_impulses = [
                        np.concatenate([
                            imp[c_min + OFDM_BODY_LENGTH:],
                            imp[:c_max],
                        ])
                    for imp in shifted_impulses
                ]
            else:
                cut_time_domain_target = -improved_time_domain_block[c_min:c_max]
                cut_shifted_impulses = [imp[c_min:c_max] for imp in shifted_impulses]

            # get rid of the imaginary parts (they should be very close to zero)
            shifted_impulses = np.array(shifted_impulses).real
            cut_shifted_impulses = np.array(cut_shifted_impulses).real
            cut_time_domain_target = cut_time_domain_target.real

            # take least squares as first-order approximation of the suppressor
            initial_coeffs, _, _, _ = scipy.linalg.lstsq(cut_shifted_impulses.T, cut_time_domain_target)

            # get a better approximation with maximum difference, which
            # is the correct cost function for maximum peak suppression

            def max_diff(x):
                return np.max(np.abs(cut_shifted_impulses.T @ x - cut_time_domain_target))
            res = scipy.optimize.minimize(max_diff, initial_coeffs)
            coeffs = res.x

            maybe_improved_block = improved_time_domain_block + shifted_impulses.T @ coeffs
            local_improvement = -cut_time_domain_target + cut_shifted_impulses.T @ coeffs

            if np.max(np.abs(maybe_improved_block)) <= np.max(np.abs(improved_time_domain_block)) \
                    and np.max(np.abs(local_improvement)) < np.max(np.abs(cut_time_domain_target)):
                # only update the block if we've actually made an improvement
                # (this method can be unstable)
                improved_time_domain_block = maybe_improved_block

    return improved_time_domain_block

def modulate_bytes(data: bytes):
    length_of_data_per_ofdm_block = (
        OFDM_DATA_INDEX_RANGE["max"] - OFDM_DATA_INDEX_RANGE["min"]
    )

    constellation_symbols = map_to_constellation_symbols(data)
    constellation_symbols = pad_symbols(
        constellation_symbols, length_of_data_per_ofdm_block
    )

    data_blocks = np.split(
        np.array(constellation_symbols),
        len(constellation_symbols) // length_of_data_per_ofdm_block,
    )

    avg_suppression = 0

    ofdm_signal = SignalBuilder()
    for block_idx, block in enumerate(data_blocks):
        # Information is only mapped to frequency bins 1 to 511.
        # Frequency bins 513 to 1023 contain the reverse ordered conjugate
        # complex values of frequency bins 1 to 511. Frequency bins 0 and 512
        # contain 0 (no information, value 0.) This all ensures that the
        # output of the OFDM modulator is a real (baseband) vector.


        fun_block = np.zeros(OFDM_DATA_INDEX_RANGE["min"] - 1)

        song_idx = 0
        for note_len, notes in SONG:
            if song_idx <= block_idx % SONG_LEN < song_idx + note_len:
                for note in notes:
                    freq = SONG_NOTES[note]
                    fun_block[get_index_of_frequency(freq)] = 1
            song_idx += note_len

        fun_block *= SONG_VOLUME * np.max(np.abs(block))


        block_with_all_freqs = np.concatenate([
            fun_block,
            block,
            np.random.choice(list(CONSTELLATION_SYMBOLS.values()), OFDM_BODY_LENGTH//2 - OFDM_DATA_INDEX_RANGE["max"])
        ])

        full_block_with_all_freqs = np.concatenate(
            [[0], block_with_all_freqs, [0], np.conjugate(block_with_all_freqs[::-1])]
        )
        assert full_block_with_all_freqs.size == OFDM_BODY_LENGTH

        time_domain_block_with_peaks = np.fft.ifft(full_block_with_all_freqs, OFDM_BODY_LENGTH)
        improved_time_domain_block = suppress_peaks(time_domain_block_with_peaks)

        if PEAK_SUPPRESSION_STATS_ENABLED:
            suppression_prc = 100 - 100 * np.max(np.abs(improved_time_domain_block))/np.max(np.abs(time_domain_block_with_peaks))
            avg_suppression += suppression_prc
            print(f"[{block_idx+1}/{len(data_blocks)}] Symbol peak suppression: {suppression_prc:.2f}%")

        block_with_cyclic_prefix = np.concatenate(
            [improved_time_domain_block[-OFDM_CYCLIC_PREFIX_LENGTH:], improved_time_domain_block]
        )

        normalized_block = block_with_cyclic_prefix.real / np.max(np.abs(block_with_cyclic_prefix.real))

        # Ensure imaginary component is zero
        assert not block_with_cyclic_prefix.imag.any()

        ofdm_signal.append_signal_part(normalized_block)

    if PEAK_SUPPRESSION_STATS_ENABLED:
        avg_suppression /= len(data_blocks)
        print(f"Average peak suppression: {avg_suppression:.2f}%", flush=True)

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

    channel_dft = np.fft.fft(channel_coefficients.real, OFDM_BODY_LENGTH)

    output_llr = []

    for block in ofdm_blocks:
        block_without_cyclic_prefix = block[OFDM_CYCLIC_PREFIX_LENGTH:]
        dft_of_block = np.fft.fft(block_without_cyclic_prefix, OFDM_BODY_LENGTH)

        equalized_dft = dft_of_block / channel_dft

        for index, noisy_symbol in enumerate(equalized_dft):
            if index < OFDM_DATA_INDEX_RANGE["min"]:
                continue  # These frequencies contain no data

            if index >= OFDM_DATA_INDEX_RANGE["max"]:
                break

            output_llr.append(noisy_symbol.real)
            output_llr.append(noisy_symbol.imag)

    return np.array(output_llr)

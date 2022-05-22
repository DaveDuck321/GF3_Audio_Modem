from config import CHIRP, CHIRP_DURATION, SAMPLE_RATE
import numpy as np

def synchronise(transmitted_signal: np.ndarray):
    # pre-crop the signal approximately to make it quicker to synchronise
    # note, TC thinks `crop_signal_into_parts` is silly because it leaves no room for error
    first_chirps = transmitted_signal[: min(3 * CHIRP.size, len(transmitted_signal)//2)]

    convolved = np.convolve(first_chirps, CHIRP[::-1])

    return synchronise_from_noisy_peaks(convolved)

def synchronise_from_noisy_peaks(signal):
    # get highest peaks
    peaks = np.sort(np.argpartition(signal, -2)[-2:])

    # check they're spaced about right
    assert abs(((peaks[1] - peaks[0]) - CHIRP_DURATION * SAMPLE_RATE)) < 0.1 * CHIRP_DURATION * SAMPLE_RATE

    # return averaged start
    return (peaks[0] + (peaks[1] - CHIRP_DURATION * SAMPLE_RATE)) // 2

def crop_signal_into_parts(transmitted_signal: np.ndarray):
    return (
        transmitted_signal[: 2 * CHIRP.size],
        transmitted_signal[2 * CHIRP.size : -2 * CHIRP.size],
        transmitted_signal[-2 * CHIRP.size :],
    )

def estimate_channel_coefficients(chirp_signal: np.ndarray):
    return np.array([1])

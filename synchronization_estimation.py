from config import CHIRP
import numpy as np


def crop_signal_into_parts(transmitted_signal: np.ndarray):
    return (
        transmitted_signal[: 2 * CHIRP.size],
        transmitted_signal[2 * CHIRP.size : -2 * CHIRP.size],
        transmitted_signal[-2 * CHIRP.size :],
    )


def estimate_channel_coefficients(chirp_signal: np.ndarray):
    return np.array([1])

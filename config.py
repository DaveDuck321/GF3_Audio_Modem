import numpy as np
from scipy import signal

from pathlib import Path


def get_index_of_frequency(f):
    return int(round(f * OFDM_BODY_LENGTH / SAMPLE_RATE))


AUDIO_SCALE_FACTOR = 0.1
SAMPLE_RATE = 48_000
MAX_RECORDING_DURATION = 120  # seconds
RECORDING_OUTPUT_DIR = Path("recordings")
TRANSMISSION_OUTPUT_DIR = Path("transmissions")

CHIRP_DURATION = 1  # Seconds
CHIRP_MIN_FREQUENCY = 1000  # Hz
CHIRP_MAX_FREQUENCY = 10_000  # Hz
CHIRP = signal.chirp(
    np.linspace(0, CHIRP_DURATION, CHIRP_DURATION * SAMPLE_RATE),
    CHIRP_MIN_FREQUENCY,
    CHIRP_DURATION,
    CHIRP_MAX_FREQUENCY,
)

OFDM_BODY_LENGTH = 1 << 13
OFDM_CYCLIC_PREFIX_LENGTH = 1 << 10
OFDM_DATA_INDEX_RANGE = {
    "min": get_index_of_frequency(1000),
    "max": get_index_of_frequency(10_000),
}

CONSTELLATION_BITS = 2
CONSTELLATION_SYMBOLS = {
    0b00: +1 + 1j,
    0b01: +1 - 1j,
    0b10: -1 + 1j,
    0b11: -1 - 1j,
}

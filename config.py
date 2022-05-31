import numpy as np
from scipy import signal

from pathlib import Path


def get_index_of_frequency(f):
    return int(round(f * OFDM_BODY_LENGTH / SAMPLE_RATE))


AUDIO_SCALE_FACTOR = 0.3
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
    "min": get_index_of_frequency(1000) + 1,
    "max": get_index_of_frequency(10_000),
}

CONSTELLATION_BITS = 2
CONSTELLATION_SYMBOLS = {
    0b00: +1 + 1j,
    0b01: +1 - 1j,
    0b10: -1 + 1j,
    0b11: -1 - 1j,
}

# Peak suppression config {{{

PEAK_SUPPRESSION_SEQUENCE = [
    # sequence of passes through peak suppression algorithm
    # peak detection threshold (in stddevs), sample view range, impulse shift range
    # this should be tweaked more!!
    (8, 50, 13),
    (8, 15, 8),
    (8, 10, 5),
    (8, 5, 3),
    (8, 4, 2),
    (6, 5, 3),
]

# }}}

SONG_VOLUME = 30
SONG = [
    (3, (392, 493.88, 587.33)),     # G4 B4 D5
    (3, (440, 554.37, 659.29)),     # A4 C#5 E5
    (2, (440,)),                    # A4
    (3, (440, 554.37, 659.29)),     # A4 C#5 E5
    (3, (493.88, 587.33, 739.99)),  # B4 D5 F#5
    (1, (880,)),                    # A5
    (1, (783.99,)),                 # G5
    (1, (739.99,)),                 # F#5
    (1, (587.33,)),                 # D5
    (3, (392, 493.88, 587.33)),     # G4 B4 D5
    (3, (440, 554.37, 659.29)),     # A4 C#5 E5
    (5, (440,)),                    # A4
    (2, tuple()),
    (1, (440,)),                    # A4
    (1, (440,)),                    # A4
    (1, (493.88,)),                 # B4
    (1, (587.33,)),                 # D5
    (2, tuple()),
    (1, (493.88,)),                 # B4
]
SONG_LEN = sum((x for x, _ in SONG))

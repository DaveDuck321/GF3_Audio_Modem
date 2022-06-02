import LDPC.ldpc as ldpc

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
    np.linspace(0, CHIRP_DURATION, int(CHIRP_DURATION * SAMPLE_RATE)),
    CHIRP_MIN_FREQUENCY,
    CHIRP_DURATION,
    CHIRP_MAX_FREQUENCY,
)


OFDM_BODY_LENGTH = 1 << 12
OFDM_CYCLIC_PREFIX_LENGTH = 1 << 9
OFDM_SYMBOL_LENGTH = OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH
OFDM_DATA_INDEX_RANGE = { # following python standard range convention
    "min": get_index_of_frequency(1000) + 1,
    "max": get_index_of_frequency(10_000) + 1,
}
MAX_NUMBER_OF_SYMBOLS_IN_FRAME = 128
KNOWN_OFDM_BLOCK_FFT = np.load('known_ofdm_symbol.npy')
KNOWN_OFDM_BLOCK = np.fft.ifft(KNOWN_OFDM_BLOCK_FFT, OFDM_BODY_LENGTH)
KNOWN_OFDM_BLOCK /= np.max(np.abs(np.fft.ifft(KNOWN_OFDM_BLOCK_FFT, OFDM_BODY_LENGTH)))
KNOWN_OFDM_REPEAT_COUNT = 4


CONSTELLATION_BITS = 2
CONSTELLATION_SYMBOLS = {
    0b00: +1 + 1j,
    0b01: +1 - 1j,
    0b10: -1 + 1j,
    0b11: -1 - 1j,
}

LDPC_CODER = ldpc.code(standard='802.16', z=64)


PEAK_SUPPRESSION_STATS_ENABLED = False
PEAK_SUPPRESSION_ENABLED = True
PEAK_SUPPRESSION_THRESH = 3.5 # stddevs

# Peak suppression stuff {{{
_perfect_time_impulse = np.zeros(OFDM_BODY_LENGTH)
_perfect_time_impulse[OFDM_BODY_LENGTH//2] = 1
_fft_impulse_approximator = np.fft.fft(_perfect_time_impulse)
_fft_impulse_approximator[OFDM_DATA_INDEX_RANGE["min"]:OFDM_DATA_INDEX_RANGE["max"]] = 0
_fft_impulse_approximator[-OFDM_DATA_INDEX_RANGE["max"]:-OFDM_DATA_INDEX_RANGE["min"]] = 0

PEAK_SUPPRESSION_IMPULSE_APPROXIMATOR = np.fft.ifft(_fft_impulse_approximator, OFDM_BODY_LENGTH)
PEAK_SUPPRESSION_SEQUENCE = [
    # sequence of passes through peak suppression algorithm
    # peak detection threshold (in stddevs), sample view range, impulse shift range
    # this should be tweaked more!!
    (PEAK_SUPPRESSION_THRESH, 50, 13),
    (PEAK_SUPPRESSION_THRESH, 15, 8),
    (PEAK_SUPPRESSION_THRESH, 10, 5),
    (PEAK_SUPPRESSION_THRESH, 5, 3),
    (PEAK_SUPPRESSION_THRESH, 4, 2),
    (PEAK_SUPPRESSION_THRESH, 5, 3),
]

# }}}

SONG_ENABLED = False
SONG_VOLUME = 30

# Song stuff {{{
SONG_NOTES = {
    "G4": 392,
    "A4": 440,
    "B4": 493.88,
    "C#5": 554.37,
    "D5": 587.33,
    "E5": 659.29,
    "F#5": 739.99,
    "G5": 783.99,
    "A5": 880,
}
_SONG = [
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4", "C#5", "E5")),
    (6, ("B4", "D5", "F#5")),
    (1, ("A5",)),
    (1, ("G5",)),
    (1, ("F#5",)),
    (1, ("D5",)),
    # --
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4",)),
    (4, tuple()),
    (1, ("A4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, tuple()),
    (1, ("B4",)),
    # --
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4", "C#5", "E5")),
    (6, ("B4", "D5", "F#5")),
    (1, ("A5",)),
    (1, ("G5",)),
    (1, ("F#5",)),
    (1, ("D5",)),
    # --
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4",)),
    (4, tuple()),
    (1, ("A4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, tuple()),
    (1, ("B4",)),
    # --
    (4, ("G4", "B4", "D5")),
    (2, ("G4", "B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("D5",)),
    (2, ("E5",)),
    (2, ("C#5",)),
    # --
    (1, ("C#5",)),
    (1, ("B4",)),
    (14, ("A4",)),
    # --
    (2, tuple()),
    (2, ("B4",)),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    (2, tuple()),
    (2, ("A4",)),
    # --
    (4, ("A5",)),
    (2, ("A5",)),
    (10, ("E5",)),
    # --
    (2, tuple()),
    (2, ("B4",)),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    (2, ("D5",)),
    (2, ("E5",)),
    # --
    (2, tuple()),
    (2, ("C#5",)),
    (2, ("B4",)),
    (1, ("C#5",)),
    (1, ("B4",)),
    (8, ("A4",)),
    # --
    (2, tuple()),
    (2, ("B4",)),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    (4, ("A4",)),
    # --
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("F#5",)),
    (8, ("E5",)),
    # --
    (10, ("D5",)),
    (2, ("E5",)),
    (2, ("F#5",)),
    (2, ("D5",)),
    # --
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("F#5",)),
    (2, ("E5",)),
    (2, ("A4",)),
    (4, ("A4",)),
    # --
    (8, tuple()),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    # --
    (2, tuple()),
    (2, ("E5",)),
    (2, ("F#5",)),
    (6, ("E5",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (2, ("F#5",)),
    (1, tuple()),
    (2, ("F#5",)),
    (1, tuple()),
    (6, ("E5",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (2, ("E5",)),
    (1, tuple()),
    (2, ("E5",)),
    (1, tuple()),
    (3, ("D5",)),
    (1, ("C#5",)),
    (2, ("B4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (4, ("D5",)),
    (2, ("E5",)),
    (3, ("C#5",)),
    (1, ("B4",)),
    (3, ("A4",)),
    (1, tuple()),
    (2, ("A4",)),
    # --
    (4, ("E5",)),
    (8, ("D5",)),
    (4, tuple()),
]

# put song into chunks
SONG = []
if SONG_ENABLED:
    while len(_SONG) > 0:
        SONG_CHUNK = []
        s = 0
        while s < 192:
            chord = _SONG.pop(0)
            SONG_CHUNK.append(chord)
            s += chord[0]
        SONG.append(SONG_CHUNK)

# TODO: this is a minor hack since we're not choosing by frame
SONG_LEN = sum((x for frame in SONG for x, _ in frame))
# }}}

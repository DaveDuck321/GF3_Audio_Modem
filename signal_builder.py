from config import MAX_RECORDING_DURATION, SAMPLE_RATE

import numpy as np


class SignalBuilder:
    def __init__(self):
        self._signal = np.empty(MAX_RECORDING_DURATION * SAMPLE_RATE)
        self._index = 0

    def append_signal_part(self, signal_part: np.ndarray):
        self._signal[self._index : self._index + signal_part.size] = signal_part
        self._index += signal_part.size

    def get_signal(self):
        return self._signal[:self._index]

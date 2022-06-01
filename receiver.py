from OFDM import demodulate_signal
from config import SAMPLE_RATE, MAX_RECORDING_DURATION, RECORDING_OUTPUT_DIR
from common import (
    save_data_to_file,
    set_audio_device_or_warn,
    finalize_argparse_for_sounddevice,
)
from error_stats import bit_error
from synchronization_estimation import (
    crop_signal_into_parts,
    estimate_channel_coefficients,
)
from ldcp_tools import decode_from_llr


import numpy as np
import sounddevice as sd
from dvg_ringbuffer import RingBuffer

import sys
from argparse import ArgumentParser


def receive_signal(signal):
    start_chirps, known_ofdm_signal, ofdm_signal, end_chirps = crop_signal_into_parts(signal)

    channel_coefficients = estimate_channel_coefficients(known_ofdm_signal)

    llr_for_each_bit = demodulate_signal(channel_coefficients, ofdm_signal)
    demodulated_data = decode_from_llr(llr_for_each_bit)

    return demodulated_data


def record_until_enter_key():
    buffer = RingBuffer(SAMPLE_RATE * MAX_RECORDING_DURATION)

    def record_callback(in_data, frames, time, status):
        buffer.extend(in_data.flatten())
        if status:
            print(status, file=sys.stderr)

    with sd.InputStream(callback=record_callback, channels=1, samplerate=SAMPLE_RATE):
        input("Press enter to stop recording")

    save_data_to_file(RECORDING_OUTPUT_DIR, buffer)
    return np.array(buffer)


if __name__ == "__main__":
    parser = ArgumentParser(description="OFDM receiver")
    parser.add_argument("file", nargs="?", help="Numpy waveform file")
    parser.add_argument("--expected_output", help="Expected transmission")
    args = finalize_argparse_for_sounddevice(parser)

    if args.file is not None:
        recorded_signal = np.load(args.file)
    else:
        set_audio_device_or_warn(args)
        recorded_signal = record_until_enter_key()

    demodulated_file = receive_signal(recorded_signal)

    if args.expected_output is not None:
        with open(args.expected_output, "rb") as expected_file:
            expected_bytes = expected_file.read()

        demodulated_file = demodulated_file[: len(expected_bytes)]
        print(
            f"[INFO] Bit error of received file: {bit_error(demodulated_file, expected_bytes)}"
        )

    open("output", "wb").write(demodulated_file)

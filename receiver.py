#  vim: set ts=4 sw=4 tw=0 et :
from OFDM import demodulate_signal
from config import OFDM_BODY_LENGTH, SAMPLE_RATE, MAX_RECORDING_DURATION, RECORDING_OUTPUT_DIR
from common import (
    save_data_to_file,
    set_audio_device_or_warn,
    finalize_argparse_for_sounddevice,
)
from error_stats import bit_error, plot_cumulative_error
from metadata import decode_received_file
from synchronization_estimation import (
    crop_frame_into_parts,
    crop_signal_into_overlapping_frames,
    estimate_channel_coefficients_and_variance,
)
from ldpc_tools import decode_from_llr


import numpy as np
import sounddevice as sd
from dvg_ringbuffer import RingBuffer

import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt


def receive_signal(signal):
    signal_frames = crop_signal_into_overlapping_frames(signal)

    llr_for_each_bit = []
    for frame in signal_frames:
        drift_per_sample, chirp_start, prefix, data, endfix, chirp_end = crop_frame_into_parts(frame)

        drift_to_endfix = drift_per_sample * (len(prefix) + len(data))

        channel_coefficients_start, normalized_variance_start = estimate_channel_coefficients_and_variance(prefix, 0, drift_per_sample)
        channel_coefficients_end, normalized_variance_end = estimate_channel_coefficients_and_variance(endfix, drift_to_endfix, drift_per_sample)

        # slight increase in performance
        channel_coefficients_mag = (np.abs(channel_coefficients_start) + np.abs(channel_coefficients_end))/2
        channel_coefficients_phase = np.angle(channel_coefficients_start)
        channel_coefficients = channel_coefficients_mag * np.exp(1j* channel_coefficients_phase)

        llr_for_each_bit.extend(demodulate_signal(channel_coefficients, data, normalized_variance_start, drift_per_sample))

    decoded_bytes = decode_from_llr(np.array(llr_for_each_bit))
    return decode_received_file(decoded_bytes)


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

    filename, demodulated_file = receive_signal(recorded_signal)

    if args.expected_output is not None:
        with open(args.expected_output, "rb") as expected_file:
            expected_bytes = expected_file.read()

        demodulated_file = demodulated_file[: len(expected_bytes)]
        print(
            f"[INFO] Bit error of received file: {bit_error(demodulated_file, expected_bytes)}"
        )
        plot_cumulative_error(demodulated_file, expected_bytes)


    open(filename, "wb").write(demodulated_file)

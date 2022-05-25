from OFDM import demodulate_signal
from config import CHIRP, OFDM_BODY_LENGTH, OFDM_CYCLIC_PREFIX_LENGTH, SAMPLE_RATE, MAX_RECORDING_DURATION, RECORDING_OUTPUT_DIR
from common import (
    save_data_to_file,
    set_audio_device_or_warn,
    finalize_argparse_for_sounddevice,
)
from synchronization_estimation import crop_signal_into_parts, estimate_channel_coefficients


import numpy as np
import sounddevice as sd
from dvg_ringbuffer import RingBuffer

import sys
from argparse import ArgumentParser


import scipy.signal
import matplotlib.pyplot as plt

def receive_signal(signal):
    start_chirps, ofdm_signal, end_chirps = crop_signal_into_parts(signal)

    plt.figure("Time domain response of channel")
    plt.title("Time domain response of channel")
    channel_coefficients_start = estimate_channel_coefficients(start_chirps)
    channel_coefficients_end = estimate_channel_coefficients(end_chirps)

    channel_coefficients_start = channel_coefficients_start / np.max(channel_coefficients_start)
    plt.plot(channel_coefficients_start)
    # plt.plot(channel_coefficients_end)

    plt.figure("Frequency response of channel")
    plt.title("Log frequency response of channel")
    plt.plot(np.log(np.fft.fft(channel_coefficients_start, OFDM_BODY_LENGTH)))
    # plt.plot(np.fft.fft(channel_coefficients_end, OFDM_BODY_LENGTH))
    plt.show()

    channel_estimates_cross_correlation = scipy.signal.correlate(channel_coefficients_start, channel_coefficients_end, mode='same')
    maximum_cross_correlation_index = np.argmax(channel_estimates_cross_correlation)

    t = np.linspace(maximum_cross_correlation_index-2, maximum_cross_correlation_index+2, 5)  # Take 5 point around the correlation peak
    quadratic_fit_of_peak = np.polyfit(t, channel_estimates_cross_correlation[maximum_cross_correlation_index-2:maximum_cross_correlation_index+3], 2)

    sampling_drift = -quadratic_fit_of_peak[1] / (2 * quadratic_fit_of_peak[0]) - OFDM_CYCLIC_PREFIX_LENGTH / 2

    

    # samples_between_chirps = CHIRP.size + ofdm_signal.size

    # samples_drifted_in_ofdm_block = (sampling_drift / samples_between_chirps) * ofdm_signal.size

    # total_length_of_signal_to_see_a_drift_of_one_sample =  ofdm_signal.size / samples_drifted_in_ofdm_block

    # new_signal = np.concatenate([ofdm_signal, [0] * round(total_length_of_signal_to_see_a_drift_of_one_sample - ofdm_signal.size)])

    # resampled_signal = scipy.signal.resample(new_signal, new_signal.size-1)

    # demodulated_signal = demodulate_signal(channel_coefficients_start, resampled_signal[:ofdm_signal.size])

    open('output', 'wb').write(demodulated_signal)
    return demodulated_signal


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
    args = finalize_argparse_for_sounddevice(parser)

    if args.file is not None:
        signal_from_file = np.load(args.file)
        receive_signal(signal_from_file)
        exit(0)

    set_audio_device_or_warn(args)

    recorded_signal = record_until_enter_key()
    receive_signal(recorded_signal)

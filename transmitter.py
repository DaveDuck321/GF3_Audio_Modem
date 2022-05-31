from config import CHIRP, SAMPLE_RATE, AUDIO_SCALE_FACTOR, TRANSMISSION_OUTPUT_DIR
from common import (
    save_data_to_file,
    set_audio_device_or_warn,
    finalize_argparse_for_sounddevice,
)
from signal_builder import SignalBuilder
import OFDM

import numpy as np
import sounddevice as sd

from argparse import ArgumentParser


def modulate_file(transmission: bytes):
    signal_builder = SignalBuilder()

    signal_builder.append_signal_part(CHIRP)
    signal_builder.append_signal_part(CHIRP)
    signal_builder.append_signal_part(OFDM.generate_known_ofdm_block())

    # TODO: Transmit duplicate OFDM block

    signal_builder.append_signal_part(OFDM.modulate_bytes(transmission))

    signal_builder.append_signal_part(CHIRP)
    signal_builder.append_signal_part(CHIRP)

    return signal_builder.get_signal()


def transmit_signal(signal):
    sd.play(AUDIO_SCALE_FACTOR * signal, samplerate=SAMPLE_RATE, blocking=True)


if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser(description="OFDM transmitter")
    parser.add_argument("file", help="File to transmit")
    parser.add_argument("--wave_file", "-w", action="store_true", help="Transmit a numpy waveform without additional processing")
    parser.add_argument("--silent", "-s", action="store_true", help="Modulate but don't play signal")
    args = finalize_argparse_for_sounddevice(parser)
    # fmt: on

    if not args.silent:
        set_audio_device_or_warn(args)

        if args.wave_file:
            signal_from_file = np.load(args.file)
            transmit_signal(signal_from_file)
            exit(0)

    with open(args.file, "rb") as input_file:
        modulated_signal = modulate_file(input_file.read())

    save_data_to_file(TRANSMISSION_OUTPUT_DIR, modulated_signal)

    if not args.silent:
        transmit_signal(modulated_signal)

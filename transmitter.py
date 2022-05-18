from config import SAMPLE_RATE, TRANSMISSION_OUTPUT_DIR
from common import set_audio_device_or_warn, finalize_argparse_for_sounddevice

import numpy as np
import sounddevice as sd

from argparse import ArgumentParser


def modulate_file(reader):
    pass


def transmit_signal(signal):
    sd.play(signal, samplerate=SAMPLE_RATE, blocking=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="OFDM transmitter")
    parser.add_argument("file", nargs="?", help="File to transmit")
    parser.add_argument("--wave_file", "-i", help="Transmit numpy waveform without additional processing")
    parser.add_argument("--silent", "-s", action="store_true", help="Modulate but don't play signal")
    args = finalize_argparse_for_sounddevice(parser)

    if not args.silent:
        set_audio_device_or_warn(args)

        if args.wave_file is not None:
            signal_from_file = np.load(args.wave_file)
            transmit_signal(signal_from_file)
            exit(0)

    with open(args.file, "rb") as input_file:
        modulated_signal = modulate_file(input_file)

    if not args.silent:
        transmit_signal(modulated_signal)

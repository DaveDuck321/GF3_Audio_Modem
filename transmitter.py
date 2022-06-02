# vim: set ts=4 sw=4 tw=0 et :
from config import (
    AUDIO_SCALE_FACTOR,
    CHIRP,
    KNOWN_OFDM_REPEAT_COUNT,
    NUMBER_OF_SYMBOLS_IN_FRAME,
    SAMPLE_RATE,
    TRANSMISSION_OUTPUT_DIR,
)

from common import (
    finalize_argparse_for_sounddevice,
    save_data_to_file,
    set_audio_device_or_warn,
    split_list_to_chunks_of_length,
)
from signal_builder import SignalBuilder
import OFDM
import ldcp_tools

import numpy as np
import sounddevice as sd

from argparse import ArgumentParser


def modulate_into_frames(known_ofdm_block, ofdm_symbols):
    """
    See standard for definitions:
        https://www.overleaf.com/project/628befa0a5f784cb3d188f72
    """
    signal_builder = SignalBuilder()

    for chunk in split_list_to_chunks_of_length(
        ofdm_symbols, NUMBER_OF_SYMBOLS_IN_FRAME["max"]
    ):
        # Preamble:
        #   single chirp
        signal_builder.append_signal_part(CHIRP)

        #   known OFDM symbols
        for _ in range(KNOWN_OFDM_REPEAT_COUNT):
            signal_builder.append_signal_part(known_ofdm_block)

        # Data:
        #   max length = 200 OFDM symbols
        for ofdm_symbol in chunk:
            signal_builder.append_signal_part(ofdm_symbol)

        #   min length = 10 OFDM symbols
        extra_symbols_needed = NUMBER_OF_SYMBOLS_IN_FRAME["min"] - len(chunk)
        if extra_symbols_needed > 0:
            for _ in range(extra_symbols_needed):
                # Fill extra "padding" symbols with the known ofdm block as a placeholder
                signal_builder.append_signal_part(known_ofdm_block)

        # End-amble
        #   known OFDM symbols
        for _ in range(KNOWN_OFDM_REPEAT_COUNT):
            signal_builder.append_signal_part(known_ofdm_block)

        # single chirp
        signal_builder.append_signal_part(CHIRP)

    return signal_builder.get_signal()


def modulate_file(file_data: bytes):
    transmission = ldcp_tools.encode_bytes(file_data)

    signal_builder = SignalBuilder()

    signal_builder.append_signal_part(CHIRP)

    known_ofdm_block = OFDM.generate_known_ofdm_block()
    all_ofdm_symbols = OFDM.modulate_bytes(transmission)

    signal_builder.append_signal_part(modulate_into_frames(known_ofdm_block, all_ofdm_symbols))

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

    print("Building OFDM symbols, please wait... ", flush=True)

    with open(args.file, "rb") as input_file:
        modulated_signal = modulate_file(input_file.read())

    save_data_to_file(TRANSMISSION_OUTPUT_DIR, modulated_signal)

    if not args.silent:
        input("Press enter to play.")
        transmit_signal(modulated_signal)

import numpy as np
import sounddevice as sd

import sys
from argparse import ArgumentParser


def record_until_enter_key():
    pass


def receive_signal(signal):
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description="OFDM receiver")
    parser.add_argument('file', nargs='?', help="Numpy waveform file")
    parser.add_argument('--device', '-d', type=int, help="Sounddevice ID to use")
    parser.add_argument('--query_devices', '-q', action="store_true", help="List sounddevices and exit")
    args = parser.parse_args()

    if args.query_devices:
        print(sd.query_devices())
        exit(0)
    
    if args.file is not None:
        signal_from_file = np.load(args.file)
        receive_signal(signal_from_file)
        exit(0)

    if args.device is not None:
        sd.default.device = args.device
    else:
        default_id = sd.default.device[0]
        default_device = sd.query_devices(default_id)
        default_api = sd.query_hostapis(default_device['hostapi'])['name']
        print(f"[INFO] no sounddevice selected, selecting: {default_id} {default_device['name']} {default_api}")
    
    recorded_signal = record_until_enter_key()    
    receive_signal(recorded_signal)

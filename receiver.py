from config import SAMPLE_RATE, MAX_RECORDING_DURATION, RECORDING_OUTPUT_DIR

import numpy as np
import sounddevice as sd
from dvg_ringbuffer import RingBuffer

import sys
import uuid
from argparse import ArgumentParser


def save_recording_to_file(recording):
    if not RECORDING_OUTPUT_DIR.exists():
        RECORDING_OUTPUT_DIR.mkdir(parents=True)

    recording_location =  RECORDING_OUTPUT_DIR / f"{uuid.uuid4().hex[:5]}.npy"
    np.save(recording_location, recording)

    print(f"[INFO] Audio recording saved to:  {recording_location}")


def record_until_enter_key():
    buffer = RingBuffer(SAMPLE_RATE * MAX_RECORDING_DURATION)
    def record_callback(in_data, frames, time, status):
        buffer.extend(in_data.flatten())
        if status:
            print(status, file=sys.stderr)

    with sd.InputStream(callback=record_callback, channels=1, samplerate=SAMPLE_RATE):
        input("Press enter to stop recording")

    save_recording_to_file(buffer)
    return buffer


def receive_signal(signal):
    sd.play(signal, samplerate=SAMPLE_RATE, blocking=True)


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

import sounddevice as sd
import numpy as np

import uuid
from argparse import ArgumentParser
from pathlib import Path


def save_data_to_file(location: Path, signal: np.ndarray):
    if not location.exists():
        location.mkdir(parents=True)

    file_location = location / f"{uuid.uuid4().hex[:5]}.npy"
    np.save(file_location, signal)

    print(f"[INFO] Signal saved to:  {file_location}")


def finalize_argparse_for_sounddevice(parser: ArgumentParser):
    parser.add_argument("--device", "-d", type=int, help="Sounddevice ID to use")
    parser.add_argument("--query_devices", "-q", action="store_true", help="List sounddevices and exit")
    args = parser.parse_args()

    if args.query_devices:
        print(sd.query_devices())
        exit(0)

    return args


def set_audio_device_or_warn(args):
    if args.device is not None:
        sd.default.device = args.device
    else:
        default_id = sd.default.device[0]
        default_device = sd.query_devices(default_id)
        default_api = sd.query_hostapis(default_device["hostapi"])["name"]
        print(
            f"[INFO] no sounddevice selected, selecting: {default_id} {default_device['name']} {default_api}"
        )

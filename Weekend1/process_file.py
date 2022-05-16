import sys
from pathlib import Path

def process_custom_file_format(input_path: str):
    file_data = open(input_path, 'rb').read()
    path, size, data = file_data.split(b'\0', 2)

    path = Path(path.decode("utf-8"))
    size = int(size.decode("utf-8"))
    data = data[:size]

    open(input_path.replace(".bin", "_") + path.name, 'wb').write(data)

process_custom_file_format(sys.argv[1])